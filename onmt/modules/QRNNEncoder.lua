--[[ QRNNEncoder is an LSTM-like quasi-RNN module used for the source language.
--]]
local QRNNEncoder, parent = torch.class('onmt.QRNNEncoder', 'nn.Container')

--[[ Construct encoder layers.

Parameters:

  * `inputNetwork` - input module.
  * `layers` - Number of QRNNEncoder layers, L.
  * `inputSize` - Size of input layer
  * `hiddenSize` - Size of the hidden layers.
  * `filterWidth` - size of the kernel size for CNN.
  * `poolingMethod` - Pooling method ['fo', 'ifo']. 'f' not supported
  * 'dropout' - Dropout rate to use.
  * `zoneout` - Zoneout rate to use.
  * `denseNet` - Use densely connected layers. (not implemented yet)
--]]


function QRNNEncoder:__init(inputNetwork, layers, inputSize, hiddenSize, filterWidth, poolingMethod, dropout, zoneout, denseNet)

  parent.__init(self)

  self.inputNet = inputNetwork

  self.args = {}
  self.args.layers = layers
  self.args.inputSize = inputSize
  self.args.outputSize = hiddenSize
  self.args.filterWidth = filterWidth
  self.args.poolingMethod = poolingMethod
  dropout = dropout or 0
  zoneout = zoneout or 0
  self.args.dropout = dropout
  self.args.zoneout = zoneout
  self.args.denseNet = denseNet

  --assert(self.args.poolingMethod == 'f' or 
  assert(self.args.poolingMethod == 'fo' or self.args.poolingMethod == 'ifo', "")

  local modules = self:_buildModel()
  self:add(modules)
end

--[[ Return a new Encoder using the serialized data `pretrained`. ]]
function QRNNEncoder.load(pretrained)
  local self = torch.factory('onmt.QRNNEncoder')()

  self.args = pretrained.args
  parent.__init(self)

  assert(#pretrained.modules == 1)
  self:add(pretrained.modules[1])

  return self
end

--[[ Return data to serialize. ]]
function QRNNEncoder:serialize()
  return {
    name = 'QRNNEncoder',
    modules = self.modules,
    args = self.args
  }
end

--[[ Move the network to train mode. ]]
function QRNNEncoder:training()
  parent.training(self)
  self.modules[1]:training()
end

--[[ Move the network to evaluation mode. ]]
function QRNNEncoder:evaluate()
  parent.evaluate(self)
  self.modules[1]:evaluate()
end

--[[ Build encoder

Returns: An nn-graph mapping

  $${
  (x_1, x_2, .. x_t, .. x_n) =>
  (c^L_1, h^L_1, c^L_2, h^L_2, .. c^L_t, h^L_t, .. c^L_n, h^L_n)
  }$$

  Where $$x_t$$ is a sparse word to lookup, and
  $$c^L$$ and $$h^L$$ are the hidden and cell states at the last layer
--]]
function QRNNEncoder:_buildModel()
--  local qrnn = nn.Sequential()
  local x = nn.Identity()() -- batchSize x seqLength
  local input = self.inputNet(x) -- batchSize x seqLength x embSize

  local outputs = {}
  local lastLayerHiddenStates

  if self.args.dropout > 0 then
    input = nn.Dropout(self.args.dropout)(input)
  end

  for i = 1, self.args.layers do
    local inputDim = ((i==1) and self.args.inputSize or self.args.outputSize)
    -- cnn
    local output = onmt.QRNNCNN.new(inputDim, self.args.outputSize, self.args.filterWidth, self.args.poolingMethod, self.args.zoneout)(input)
    -- batchSize x seqlength x (hiddenSize * numGates)
    -- rnn
    output = nn.Transpose({1,2})(output) -- DoubleTensor: seqlength x batchSize x (hiddenSize * numGates)
    local stepmodule = nn.Sequential() -- applied at each time-step
    stepmodule:add(onmt.QRNNRNN.new(self.args.outputSize, self.args.poolingMethod))
    if self.args.dropout > 0 then stepmodule:add(nn.Dropout(self.args.dropout)) end
    output = nn.Sequencer(stepmodule)(output) -- DoubleTensor: seqlength x batchSize x hiddenSize
    lastLayerHiddenStates = nn.Transpose({1,2})(output) -- DoubleTensor: batchSize x seqlength x hiddenSize
    local finalTimeStep = nn.Select(1,-1)(output) -- DoubleTensor: batchSize x hiddenSize
    table.insert(outputs, finalTimeStep)
    input = lastLayerHiddenStates
  end

  table.insert(outputs, lastLayerHiddenStates)
  return nn.gModule({x}, outputs)
end

--[[Compute the context representation of an input.

Parameters:

  * `batch` - as defined in batch.lua.

Returns:

  1. - final hidden states
  2. - context matrix H (batch.size, batch.sourceLength, outputSize)
--]]
function QRNNEncoder:forward(batch)
  local x = batch:getSourceInput() -- {batch size x sen length}
  print('Input x[1]:size(): ' .. tostring(x[1]:size()))
  local finalStates = self.modules[1]:forward(x)
  local context = table.remove(finalStates, #finalStates)

  print('#finalStates: ' .. tostring(#finalStates))
  print('finalStates[1]: ' .. tostring(finalStates[1]:size()))
  print('context: ' .. tostring(context:size())) -- should be batchSize x sourceLength x rnnSize
  
  return finalStates, context
end

--[[ Backward pass (only called during training)

  Parameters:

  * `batch` - must be same as for forward
  * `gradStatesOutput` gradient of loss wrt last state - this can be null if states are not used
  * `gradContextOutput` - gradient of loss wrt full context.

  Returns: `gradInputs` of input network.
--]]
function QRNNEncoder:backward(batch, gradStatesOutput, gradContextOutput)
  local gradInputs = self.modules[1]:backward({batch}, {gradStatesOutput, gradContextOutput})
  return gradInputs
end
