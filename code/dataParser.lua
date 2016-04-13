require 'torch'

-- local filePath = './nips13-dataset/Freebase/train.txt'
local filePath = './nips13-dataset/Wordnet/train.txt'
-- local filePath = './testTrain.txt'
local fileIter = io.open(filePath, 'r')

function string:split(sep)
	local sep, fields = sep, {}
	local pattern = string.format("([^%s]+)", sep)
	self:gsub(pattern, function(substr) fields[#fields + 1] = substr end)
	return fields
end

-- class = {}
entityRelation = {}

local i = 0
for line in fileIter:lines('*l') do
	i = i + 1
	entityRelation[i]={}
	local l = line:split('	')
	for key, val in ipairs(l) do
		entityRelation[i][key] = val
	end
end
fileIter:close()

local filePath = './nips13-dataset/Wordnet/test.txt'
local fileIter = io.open(filePath, 'r')
i=0
testData = {}
testClass = {}
for line in fileIter:lines('*l') do
	i = i + 1
	testData[i]={}
	local l = line:split('	')
	for key, val in ipairs(l) do
		if key == 4 then
			testClass[i] = val
		else
			testData[i][key] = val
		end
	end
end
fileIter:close()

-----------------------------------------------------------------------------------------------------
--------------------------------		Define vocab map  		-------------------------------------
-----------------------------------------------------------------------------------------------------

vocab = {}
relation = {}
dataset = {}
mapID = 0
relationID = 0
dataset = {}
relationVocab = {}
ent1Freq = {}
word_embed_size = 100
ent2Freq = {}
relationEntVocab = {}

for i=1, #entityRelation do
	for j = 1,3 do
		-- add words to vocabulary if not already there
		if vocab[entityRelation[i][j]] == nil then
			mapID = mapID + 1;
			vocab[entityRelation[i][j]] = mapID;
		end
	end
	if relation[entityRelation[i][2]] == nil then
			relationID = relationID + 1;
			-- relation[] stores index of a relation
			relation[entityRelation[i][2]] = relationID;
			-- relationVocab stores entities corresponding to that relation index
			relationVocab[relationID] = {}
			relationVocab[relationID][1] = entityRelation[i][1]
			relationEntVocab[relationID] = {}
			relationEntVocab[relationID][1] = {}
			relationEntVocab[relationID][1][1] = entityRelation[i][3]
			-- ent1Freq stores no of entries corresponding to a particular relationID
			ent1Freq[relationID] = 1
			ent2Freq[relationID] = {}
			ent2Freq[relationID][ent1Freq[relationID]] = 1
	else
		local tempID = relation[entityRelation[i][2]]
		local flag = 1
		for j = 1,ent1Freq[tempID] do
			if relationVocab[tempID][j] == entityRelation[i][1] then
				-- add ent2 corresponding to ent1,relation pair
				ent2Freq[tempID][j] = ent2Freq[tempID][j] + 1
				relationEntVocab[tempID][j][ent2Freq[tempID][j]] = entityRelation[i][3]
				flag = 0
			end
		end
		if flag == 1 then
			ent1Freq[tempID] = ent1Freq[tempID] + 1
			relationVocab[tempID][ent1Freq[tempID]] = entityRelation[i][1]
			relationEntVocab[tempID][ent1Freq[tempID]] = {}
			ent2Freq[tempID][ent1Freq[tempID]] = 1
			relationEntVocab[tempID][ent1Freq[tempID]][1] = entityRelation[i][3]
		end
	end
	local entity1 = vocab[entityRelation[i][1]]
	local entity2 = vocab[entityRelation[i][3]]
	local tempInput = torch.Tensor{entity1, entity2}
	-- local tempInput = {torch.Tensor{entity1}, torch.Tensor{entity2}}
	-- print(tempInput)

	-- output are relation labels for training of word2vec clusters
	local tempOutput = torch.Tensor{relation[entityRelation[i][2]]}
	dataset[i] = {tempInput,tempOutput}
	-- dataset[i][1] -> two entities
	-- dataset[i][2] -> relation between them
end

print(relationID)
local s = 0
for i = 1,relationID do
	-- print(ent1Freq[i])
	for j = 1,ent1Freq[i] do
		-- print(ent2Freq[i][j])
		s = s + ent2Freq[i][j]
	end
	-- print(s)
	-- s = 0
end
-- print(s)
-- print(#entityRelation)