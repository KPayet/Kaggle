io.output("predictions.txt")
for i=1,testData:size() do
    pred = model:forward(testData.data[i]:cuda())
    for j=1,10 do
        if pred[j]==pred:max() then
            print(j)
        end
    end
end
