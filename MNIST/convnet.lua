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

if not os.execute("ls train.csv") then
    os.execute("aws s3 cp s3://kpayets3/Data/Kaggle/MNIST/train.csv .")
    os.execute("sed '1d' train.csv > tmpfile; mv tmpfile train.csv")
end
if not os.execute("ls test.csv") then
    os.execute("aws s3 cp s3://kpayets3/Data/Kaggle/MNIST/test.csv .")
    os.execute("sed '1d' test.csv > tmpfile; mv tmpfile test.csv")
end

train = csvigo.load{path = "train.csv", mode="raw", header=false}
train = torch.Tensor(train)
test = csvigo.load{path = "test.csv", mode="raw", header=false}
test = torch.Tensor(test)

trainData = {
    
    data = train[{{},{2,785}}],
    labels = train[{{}, 1}],
    size = function() return (#trainData.data)[1] end
}
testData = {
    data = test[{{},{1,784}}],
    --labels = test[{{}, 1}],
    size = function() return (#testData.data)[1] end
}

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
-- input back to image
model:add(nn.Reshape(1,28,28))

-- 1st conv layer
model:add(nn.SpatialDropout(0.1))
model:add(nn.SpatialConvolutionMM(1,16,5,5,1,1,2))
model:add(cudnn.ReLU())
model:add(nn.SpatialMaxPooling(2,2))

-- 2nd conv layer
model:add(nn.SpatialDropout(0.5))
model:add(nn.SpatialConvolutionMM(16,256,5,5,1,1,1))
model:add(cudnn.ReLU())
model:add(nn.SpatialMaxPooling(2,2))

-- 3rd conv layer
model:add(nn.SpatialDropout(0.5))
model:add(nn.SpatialConvolutionMM(256,2048,5,5))
model:add(cudnn.ReLU())
model:add(nn.SpatialMaxPooling(2,2))

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
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+batchSize-1,maxEntries) do
	 -- load new sample
	 local input = trainData.data[shuffle[i]]:double()
	 local target = trainData.labels[shuffle[i]]
	 table.insert(inputs, input)
	 table.insert(targets, target)
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
		       for i = 1,#inputs do
			  -- estimate f
			  local output = model:forward(inputs[i]:cuda())
			  output = output:double()
			  if targets[i]==0 then targets[i]=10 end
			  
			  local err = criterion:forward(output, targets[i])
			  f = f + err
			  
			  -- estimate df/dW
			  local df_do = criterion:backward(output, targets[i])
			  model:backward(inputs[i]:cuda(), df_do:cuda())

			  -- update confusion
			  confusion:add(output, targets[i])
		       end
			
		       -- normalize gradients and f(X)
		       gradParameters:div(#inputs)
		       f = f/#inputs

		       -- return f and df/dX
		       return f,gradParameters
		    end

	config = config or {learningRate = 1e-3,
			 weightDecay = 0,
			 momentum = 0,
			 learningRateDecay = 5e-7}
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
   local filename = paths.concat('./mnist_convnet_model_big.net')
   
   print('==> saving model to '..filename)
   torch.save(filename, model)

   os.execute("aws s3 cp ./mnist_convnet_model_big.net s3://kpayets3/mnist_convnet_model_big.net")

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

