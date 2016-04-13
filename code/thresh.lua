require 'torch'
require 'nn'
require 'optim'
require 'dp'
dofile('./dataParser.lua')


local filePath = './nips13-dataset/Wordnet/dev.txt'
local fileIter = io.open(filePath, 'r')
local it=0
testData = {}
testClass = {}
for line in fileIter:lines('*l') do
	it = it + 1
	testData[it]={}
	local l = line:split('	')
	for key, val in ipairs(l) do
		if key == 4 then
			testClass[it] = val
		else
			testData[it][key] = val
		end
	end
end
fileIter:close()

word2vec = torch.load("word2vec100.net")

correct = {}
c = 0
sc = 0
cmax = -100
cmin = 100

incorrect = {}
inc = 0
sinc = 0
incmax = -100
incmin = 100

-- print(#testClass)
newEntities = 0

for i = 1 ,#testClass do
	local word_embed_size = 100
	if relation[testData[i][2]] == nil or vocab[testData[i][1]] == nil or vocab[testData[i][3]] == nil then
		newEntities = newEntities + 1;
		goto continue
	end
	local inputs = torch.Tensor{vocab[testData[i][1]], vocab[testData[i][3]]}
    local ent = word2vec:forward(inputs)
    -- print(ent)
    local ent1 = torch.Tensor(1,word_embed_size)
    local ent2 = torch.Tensor(1,word_embed_size)
    for j = 1,100 do
		ent1[1][j] = ent[j]
		ent2[1][j] = ent[word_embed_size+j]
    end
    local inputVector = {ent1, ent2}
	target = testClass[i]
	local relationID = relation[testData[i][2]]
	model = torch.load("final_model" .. relationID .. ".net")
	testOutput = model:forward(inputVector)
	
	if target == '1' then
		c = c + 1
		correct[c] = testOutput[1]
		if testOutput[1] > cmax then
			cmax = testOutput[1]
		end
		if testOutput[1] < cmin then
			cmin = testOutput[1]
		end
		sc = sc + testOutput[1]
	else
		inc = inc + 1
		incorrect[inc] = testOutput[1]
		if testOutput[1] > incmax then
			incmax = testOutput[1]
		end
		if testOutput[1] < incmin then
			incmin = testOutput[1]
		end
		sinc = sinc + testOutput[1]
	end
	::continue::
end

print(newEntities)

print("correct")
print("max->", cmax)
print("min->", cmin)
print("avg-> ", sc/c)

print("incorrect")
print("max->", incmax)
print("min->", incmin)
print("avg-> ", sinc/inc)