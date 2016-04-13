require 'torch'
require 'nn'
require 'optim'
require 'dp'
dofile('./dataParser.lua')

-- Define constants
local vocab_size = mapID -- number of words in the vocabulary
local word_embed_size = 100 -- size of word embedding you are looking for
local learning_rate = 0.01 -- initial learning rate for the training
local window_size = 2 -- no. of surrounding words to predict. 2 means left and right word
local epochs = 5 -- number of complete passes of the training set


k = 4
y = torch.Tensor(k)

-- Define your model

model = nn.ConcatTable()

submodel1 = nn.Sequential()
submodel1:add(nn.Bilinear(word_embed_size, word_embed_size, k))
submodel1:add(nn.View(k))

submodel2 = nn.ParallelTable()

sub1 = nn.Linear(word_embed_size, k)
sub2 = nn.Linear(word_embed_size, k)

submodel2:add(sub1)
submodel2:add(sub2)
model2 = nn.Sequential()
model2:add(submodel2)
model2:add(nn.CAddTable())
model2:add(nn.View(k))

model:add(submodel1)
model:add(model2)

final_model = nn.Sequential()
final_model:add(model)
final_model:add(nn.CAddTable())
final_model:add(nn.Tanh())
final_model:add(nn.Linear(k, 1, false))


criterion = nn.MarginCriterion(1)
-- print(vector_model:forward( tempInput[1]))

word2vec = torch.load("word2vec100.net")

x, dl_dx = final_model:getParameters()

feval = function(x_new)
   if x ~= x_new then
      x:copy(x_new)
   end

   -- reset gradients
   dl_dx:zero()

   -- and batch over the whole training dataset:
   local loss_x = 0

   -- print(inputs[2])
   for i = 1,#inputs do
      local ent = word2vec:forward(inputs[i])
      local ent1 = torch.Tensor(1,word_embed_size)
      local ent2 = torch.Tensor(1,word_embed_size)
      -- print(ent[200])
      for j = 1,100 do
         ent1[1][j] = ent[j]
         ent2[1][j] = ent[word_embed_size+j]
      end
      local inputVector = {ent1, ent2}
      local target = torch.Tensor{1}
      -- local op = final_model:forward(inputVector)
      
      -- evaluate the loss function and its derivative wrt x, for that sample
      loss_x = loss_x + criterion:forward((final_model:forward(inputVector)), target)
      final_model:backward(inputVector, criterion:backward((final_model.output), target))
   end
   for i = 1,10 do
      local ent = word2vec:forward(corruptedEntries[i])
      local ent1 = torch.Tensor(1,word_embed_size)
      local ent2 = torch.Tensor(1,word_embed_size)
      for j = 1,100 do
         ent1[1][j] = ent[j]
         ent2[1][j] = ent[word_embed_size+j]
      end
      local inputVector = {ent1, ent2}
      local target = torch.Tensor{-1}
      loss_x = loss_x + criterion:forward((final_model:forward(inputVector)), target)
      final_model:backward(inputVector, criterion:backward((final_model.output), target))
   end

   -- normalize with batch size
   loss_x = loss_x / (#dataset)
   dl_dx = dl_dx:div( (#dataset) )

   -- return loss(x) and dloss/dx
   return loss_x, dl_dx
end

-- L-BFGS parameters are different than SGD:
--   + a line search: we provide a line search, which aims at
--                    finding the point that minimizes the loss locally
--   + max nb of iterations: the maximum number of iterations for the batch,
--                           which is equivalent to the number of epochs
--                           on the given batch. In that example, it's simple
--                           because the batch is the full dataset, but in
--                           some cases, the batch can be a small subset
--                           of the full dataset, in which case maxIter
--                           becomes a more subtle parameter.

lbfgs_params = {
   -- lineSearch = optim.lswolfe,
   -- maxIter = epochs,
   verbose = true
}

config = {
   learningRate = 0.01,
   learningRateDecay = 0,
   weightDecay = 0.001,
   momentum = 0.2,
   nesterov = true,
   dampening = 0
}
state = config

print('============================================================')
print('Training with L-BFGS')

for i = 1, relationID do
   final_model:reset()
   for epoch = 1,epochs do
      for j = 1, ent1Freq[i] do
         
         inputs = {}
         for p = 1, ent2Freq[i][j] do
            inputs[p] = torch.Tensor{vocab[relationVocab[i][j]], vocab[relationEntVocab[i][j][p]]}
         end
         
         corruptedEntries = {}
         local corruptCount = 0
         while corruptCount < 10 do
            local ent1ID = (torch.random()%ent1Freq[i]) + 1
            if ent1ID ~= j then
               corruptCount = corruptCount + 1
               local ent2ID = (torch.random()%ent2Freq[i][ent1ID]) + 1
               local flag = 0
               for q = 1,ent2Freq[i][ent1ID] do
                  if relationEntVocab[i][ent1ID][ent2ID] == relationEntVocab[i][j][q] then
                     flag = 1
                     break
                  end
               end
               if flag == 1 then
                  corruptCount = corruptCount - 1
               else
                  corruptedEntries[corruptCount] = torch.Tensor{vocab[relationVocab[i][j]], vocab[relationEntVocab[i][ent1ID][ent2ID]]}
               end
            end
         end

         _,fs = optim.lbfgs(feval,x,lbfgs_params)
         -- _,fs = optim.sgd(feval,x,config,state)
         print('loss = ' .. fs[1] .. ' for (entity1, relation) -> ('.. j .. ', ' .. i .. ') for epoch = ' .. epoch)
      end
   end
   torch.save("final_model".. i ..".net",final_model)
end