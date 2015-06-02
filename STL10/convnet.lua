-- Simple supervised learning on MNIST, using a conv net 
-- Uses the h2o training version of mnist
-- This is a one file script, no options, just to keep it simple

-- Loading data

require 'os'
require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'csvigo'
require 'optim'

matio = require 'matio'

train = matio.load("./stlTrainSubset.mat")
trainData = {
    data = train.trainImages,
    labels = train.trainLabels[{{},1}],
    size = function() return (#trainData.data)[1] end
}
trainData.data = trainData.data:transpose(1,4):transpose(2,3):transpose(3,4)

test = matio.load("./stlTestSubset.mat")
testData = {
    data = test.testImages,
    labels = test.testLabels[{{},1}],
    size = function() return (#testData.data)[1] end
}
testData.data = testData.data:transpose(1,4):transpose(2,3):transpose(3,4)

-- Normalize features globally

mean = trainData.data:mean()
std = trainData.data:std()

trainData.data:add(-mean)
trainData.data:div(std)

testData.data:add(-mean)
testData.data:div(std)

noutputs = 10

-- Init GPU

print(  cutorch.getDeviceProperties(cutorch.getDevice()) )

-- Build model

model = nn.Sequential()

-- 1st conv layer
model:add(nn.SpatialConvolutionMM(3,48,11,11,1,1,0))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialMaxPooling(2,2))

-- 2nd conv layer
--model:add(nn.SpatialDropout(0.1))
model:add(nn.SpatialConvolutionMM(48,256,10,10,1,1,0))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialMaxPooling(2,2))

-- 3rd conv layer
--model:add(nn.SpatialDropout(0.1))
model:add(nn.SpatialConvolutionMM(256,1024,8,8))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialMaxPooling(2,2))

-- 4th conv layer
model:add(nn.SpatialConvolutionMM(1024,2048,6,6))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialMaxPooling(2,2))

-- 5th conv layer
model:add(nn.SpatialConvolutionMM(2048,2048,5,5))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialMaxPooling(2,2))

-- 6th conv layer
model:add(nn.SpatialConvolutionMM(2048,2048,3,3))
model:add(cudnn.ReLU())

model:add(nn.Reshape(2048))
model:add(nn.Dropout(0.5))

-- Full connected ff net
model:add(nn.Linear(2048, 1024))
model:add(cudnn.ReLU())
model:add(nn.Dropout(0.5))

model:add(nn.Linear(1024, 512))
model:add(cudnn.ReLU())
model:add(nn.Dropout(0.5))

model:add(nn.Linear(512, 128))
model:add(cudnn.ReLU())
model:add(nn.Dropout(0.5))

--Output layer
model:add(nn.Linear(128, noutputs))
model:add(nn.LogSoftMax()) -- needed for NLL criterion

model:cuda()

--Loss function

criterion = nn.ClassNLLCriterion()
criterion:cuda()

-- Training -- This part is an almost copy/paste of http://code.madbits.com/wiki/doku.php?id=tutorial_supervised_4_train

classes = {'1','2','3','4','5','6','7','8','9','0'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- Log results to files
trainLogger = optim.Logger('./h2o_convnet_train.log')
testLogger = optim.Logger('./h2o_convnet_test.log')

if model then
    parameters,gradParameters = model:getParameters()
end

trsize = trainData:size()
batchSize = 128

-- Training function
function train(maxEntries)
   
   local maxEntries = maxEntries or trainData:size()   

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()
   
   model:training()
    
   -- shuffle at each epoch
   shuffle = torch.randperm(maxEntries)

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch)
   for t = 1,maxEntries,batchSize do
      -- disp progress
      xlua.progress(t, maxEntries)

      -- create mini batch
      local inputs = torch.Tensor(batchSize,784)
      local targets = torch.Tensor(batchSize)
      
      local k = 1
      for i = t,math.min(t+batchSize-1,trainData:size()) do
         -- load new sample
         local sample = trainData.data[i]
         local input = trainData.data[i]
         local target = trainData.labels[i]
         inputs[k] = input
	 if target==0 then target=10 end
         targets[k] = target
         k = k + 1
      end      
      
      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
		       -- get new parameters
		       if x ~= parameters then
			  parameters:copy(x)
		       end

		       -- reset gradients
		       gradParameters:zero()

		       -- f is the average of all criterions
		       local f = 0
                       -- evaluate function for complete mini batch
		       inputs = inputs:cuda()
	 	       targets = targets:cuda()
         	       local outputs = model:forward(inputs)
         	       --outputs = outputs:double()
		       local f = criterion:forward(outputs, targets)

         	       -- estimate df/dW
                       local df_do = criterion:backward(outputs, targets)
        	       model:backward(inputs, df_do)
                       
                                -- update confusion
         	       for i = 1,batchSize do
                       	  confusion:add(outputs[i], targets[i])
                       end	
			
		       -- return f and df/dX
		       return f,gradParameters
    end

	config = config or {learningRate = 1e-4,
			 weightDecay = 0,
			 momentum = 0,
			 learningRateDecay = 0}
	optim.sgd(feval, parameters, config)
	

   end

   -- time taken
   time = sys.clock() - time
   time = time / maxEntries
   print("==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- update logger/plot
   --trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   
   -- print confusion matrix
   print(confusion)
   confusion:zero()

   -- save/log current net
   local filename = paths.concat('./mnist_convnet_model_big_2.net')
   
   print('==> saving model to '..filename)
   torch.save(filename, model)

   os.execute("aws s3 cp ./mnist_convnet_model_big_2.net s3://kpayets3/mnist_convnet_model_big_2.net")

   -- next epoch
   epoch = epoch + 1
end

function test(maxEntries)
   
   local maxEntries = maxEntries or testData:size()
   
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end
   
   model:evaluate()

   -- test over test data
   print('==> testing on test set:')
   for t = 1,maxEntries do
      -- disp progress
      xlua.progress(t, maxEntries)

      -- get new sample
      local input = testData.data[t]:double()
      local target = testData.labels[t]
      if target == 0 then target = 10 end

      -- test sample
      local pred_gpu = model:forward(input:cuda())
      local pred = pred_gpu:double()
      confusion:add(pred, target)
   end

   -- timing
   time = sys.clock() - time
   time = time / maxEntries
   print("==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}

   -- print confusion matrix
   print(confusion)
   confusion:zero()
end

--[[epoch=1
while epoch<11 do
   train()
   test()
end--]]

