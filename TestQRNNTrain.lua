require('torch')
require('rnn')
require('nngraph')
require('optim')
require('TestQRNNRNN')

function buildQRNNCNN(inputSize, hiddenSize, filterWidth, poolingMethod, zoneout)
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


function buildQRNN(vocabSize, embSize, layers, hiddenSize, filterWidth, poolingMethod, zoneout, dropout)

  local qrnn = nn.Sequential()

  local lookup = nn.LookupTable(vocabSize, embSize) -- batchSize x seqLength => batchSize x seqLength x embSize
  qrnn:add(lookup)
  if dropout > 0 then
    qrnn:add(nn.Dropout(dropout))
  end

  for i=1,layers do
    local inputDim = ((i==1) and embSize or hiddenSize)
    -- cnn
    qrnn:add(buildQRNNCNN(inputDim, hiddenSize, filterWidth, poolingMethod, zoneout)) -- batchSize x seqlength x (hiddenSize * numGates)
    qrnn:add(nn.Transpose({1,2})) -- DoubleTensor: seqlength x batchSize x (hiddenSize * numGates)
    -- rnn
    local stepmodule = nn.Sequential() -- applied at each time-step
    stepmodule:add(TestQRNNRNN.new(hiddenSize, poolingMethod))
    if dropout > 0 then stepmodule:add(nn.Dropout(dropout)) end
    qrnn:add(nn.Sequencer(stepmodule)) -- DoubleTensor: seqlength x batchSize x hiddenSize
    local transposer = nn.Transpose({1,2})
    transposer.name = 'QRNN_RNN'
    qrnn:add(transposer) -- DoubleTensor: batchSize x seqlength x hiddenSize
  end

  return qrnn
end

local model = buildQRNN(--[[vocabSize--]]17,
                        --[[embSize--]]5,
                        --[[layers--]]2,
                        --[[hiddenSize--]]7,
                        --[[filterWidth--]]2,
                        --[[poolingMethod--]]'ifo',
                        --[[zoneout--]]0.1,
                        --[[dropout--]]0.1)

local QRNN_RNNs = {}

model:apply(function (layer)
      if layer.name == 'QRNN_RNN' then
        table.insert(QRNN_RNNs, layer)
      end
    end)

local input = torch.Tensor(6, 3):random(1,17)

print('Input: ')
print(input) -- batchSize x  seqLength

local output = model:forward(input)

print('Output: ')
print(output) -- batchSize x seqLength x hiddenSize

print('QRNN_RNN.output')
for i=1,#QRNN_RNNs do
  print(QRNN_RNNs[i].output[{{},3,{}}])
end

local fakeRef = output:clone():random(1,17)

-- local err = criterion:forward(output, fakeRef)
-- local gradOutputs = criterion:backward(outputs, targets)
-- local gradInputs = model:backward(inputs, gradOutputs)

