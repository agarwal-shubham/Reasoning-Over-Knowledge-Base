require 'torch'
require 'nn'
require 'optim'
require 'dp'
dofile('./dataParser.lua')

word2vec = torch.load("word2vec100.net")
-- model = torch.load("final_model.net")
testAccuracy = 0
max = -100
min = 100
cmax = -100
cmin = 100
s = 0
for i = 1 ,#entityRelation do
	local word_embed_size = 100
	local inputs = dataset[i][1]
    local ent = word2vec:forward(inputs)
    local ent1 = torch.Tensor(1,word_embed_size)
    local ent2 = torch.Tensor(1,word_embed_size)
    for j = 1,100 do
		ent1[1][j] = ent[j]
		ent2[1][j] = ent[word_embed_size+j]
    end
    local inputVector = {ent1, ent2}
	target = 1
	local relationID = relation[entityRelation[i][2]]
	model = torch.load("final_model" .. relationID .. ".net")
	testOutput = model:forward(inputVector)
	-- print(testOutput)
	-- print(relation[target])
	-- relation should be in top 10 percent of the output
	thresh = 0.5
	-- print(thresh)
	if testOutput[1] > max then
		max = testOutput[1]
	end
	if testOutput[1] < min then
		min = testOutput[1]
	end
	s = s + testOutput[1]
	if testOutput[1] >= thresh then
		testAccuracy = testAccuracy + 1;
		if testOutput[1] > cmax then
			cmax = testOutput[1]
		end
		if testOutput[1] < cmin then
			cmin = testOutput[1]
		end
	end
end
print("max->", max)
print("min->", min)
print("cmax->", cmax)
print("cmin->", cmin)
print("avg-> ", s/(#entityRelation))
-- print(testAccuracy)
testAccuracy = testAccuracy/(#entityRelation)
print(testAccuracy)
