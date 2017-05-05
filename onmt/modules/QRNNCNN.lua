require('nngraph')

--[[
Implementation of QRNN's CNN step as an nn unit.

Computes $$(x_1, x_2 .. x_t, .. x_n) => (Z, F, O)$$.

--]]
local QRNNCNN, parent = torch.class('onmt.QRNNCNN', 'onmt.Network')

--[[
Parameters:

  * `layers` - Number of LSTM layers, L.
  * `inputSize` - Size of input layer
  * `hiddenSize` - Size of the hidden layers.
  * `filterWidth` - size of the kernel size for CNN.
  * `poolingMethod` - Pooling method ['f', 'fo', 'ifo'].
  * `zoneout` - Zoneout rate to use.
--]]
function QRNNCNN:__init(inputSize, hiddenSize, filterWidth, poolingMethod, zoneout)

  self.inputSize = inputSize
  self.outputSize = hiddenSize
  self.zoneout = zoneout or 0

  local net = self:_buildModel(inputSize, hiddenSize, filterWidth, poolingMethod, zoneout)
  parent.__init(self, net)
end

--[[ Stack the LSTM units. ]]
function QRNNCNN:_buildModel(inputSize, hiddenSize, filterWidth, poolingMethod, zoneout)
  local numGates
  if poolingMethod == 'f' then
    numGates = 2
  elseif poolingMethod == 'fo' then
    numGates = 3
  elseif poolingMethod == 'ifo' then
    numGates = 4
  else
    print('QRNNCNN: Invalid poolingMethod: ' .. poolingMethod)
    os.exit()
  end

  local input = nn.Identity()() -- x: batchSize x seqLength x inputSize

  local conv = nn.Padding(--[[dim]] 1, --[[pad]] -(filterWidth - 1), --[[nInputDim]] 2, --[[value]] 0)(input)
  conv = nn.TemporalConvolution(--[[inputFrameSize]] inputSize, --[[outputFrameSize]] hiddenSize * numGates, --[[kW]] filterWidth, --[[dW]] 1)(conv)
  -- batchSize x T x (hiddenSize * numGates)

  local outputs = {}
  local Z = nn.Tanh()(nn.Narrow(3, 1 + 0 * hiddenSize, hiddenSize)(conv))
  table.insert(outputs, Z)

  local F = nn.Sigmoid()(nn.Narrow(3, 1 + 1 * hiddenSize, hiddenSize)(conv))
  -- zoneout: F = 1 - dropout(1 - sigmoid(W_f * X))
  if zoneout > 0 then
    F = nn.AddConstant(1, true)(nn.MulConstant(-1, true)(nn.Dropout(zoneout)(nn.AddConstant(1, true)(nn.MulConstant(-1, true)(F)))))
  end
  table.insert(outputs, F)

  if numGates >= 3 then
    local O = nn.Sigmoid()(nn.Narrow(3, 1 + 2 * hiddenSize, hiddenSize)(conv))
    table.insert(outputs, O)
  end

  if numGates >= 4 then
    local I = nn.Sigmoid()(nn.Narrow(3, 1 + 3 * hiddenSize, hiddenSize)(conv))
    table.insert(outputs, I)
  end

  return nn.gModule({input}, {nn.JoinTable(2,2)(outputs)})
end
