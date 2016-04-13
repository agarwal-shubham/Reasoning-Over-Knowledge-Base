require 'torch'
require 'dp'
require 'nn'
require 'optim'
dofile('./dataParser.lua')

-- Define constants
local vocab_size = mapID -- number of words in the vocabulary
local learning_rate = 0.1 -- initial learning rate for the training
local window_size = 2 -- no. of surrounding words to predict. 2 means left and right word
local epochs = 5 -- number of complete passes of the training set
-- print(vocab_size)
-- print(#entityRelation)

-- Prepare your dataset
function dataset:size() return #entityRelation end -- define the number of input samples in your dataset (which is 2)

-- Define your model
word2vec = nn.Sequential()
word2vec:add(nn.LookupTable(vocab_size, word_embed_size)) -- consumes the word indices and outputs the embeddings
-- word2vec:add(nn.Collapse(2))
word2vec:add(nn.Mean())
word2vec:add(nn.Linear(word_embed_size, vocab_size))
word2vec:add(nn.LogSoftMax())


-- Define the loss function (Negative log-likelihood)
criterion = nn.ClassNLLCriterion()
trainer = nn.StochasticGradient(word2vec, criterion)
trainer.learningRate = learning_rate

-- print(word2vec:forward(dataset[1][1]))
trainer:train(dataset)
-- print(word2vec:forward(dataset[1][1]))

torch.save("word2vec100.net",word2vec)