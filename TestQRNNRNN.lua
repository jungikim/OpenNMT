
-- modified from https://raw.githubusercontent.com/Element-Research/rnn/master/LSTM.lua
-- to make it work with nn.Sequencer

require('rnn')
require('nngraph')

local TestQRNNRNN, parent = torch.class('TestQRNNRNN', 'nn.AbstractRecurrent')

function TestQRNNRNN:__init(outputSize, poolingMethod, rho)
   parent.__init(self, rho or 9999)
   self.outputSize = outputSize
   self.poolingMethod = poolingMethod

   self.recurrentModule = self:buildModel()
   -- make it work with nn.Container
   self.modules[1] = self.recurrentModule
   self.sharedClones[1] = self.recurrentModule

   -- for output(0), cell(0) and gradCell(T)
   self.zeroTensor = torch.Tensor()

   self.cells = {}
   self.gradCells = {}
end

-- input of Model is table: {input, cell(t-1)}
-- output of Model is table : {output(t), cell(t)}
function TestQRNNRNN:buildModel()
  local numGates
--  if self.poolingMethod == 'f' then        numGates = 2
--  else
  if self.poolingMethod == 'fo' then   numGates = 3
  elseif self.poolingMethod == 'ifo' then  numGates = 4
  else
    print('QRNNCNN: Invalid poolingMethod: ' .. self.poolingMethod)
    os.exit()
  end

  local ZFOI = nn.Identity()() -- batchSize x inputSize
--  local prevH = nn.Identity()() -- batchSize x inputSize or nil if numGates != 2
  local prevC = nn.Identity()() -- batchSize x inputSize or nil if numGates >= 3

  local z, f, o, i
  z = nn.Narrow(2, 1 + 0 * self.outputSize, self.outputSize)(ZFOI) -- batchSize x inputSize
  f = nn.Narrow(2, 1 + 1 * self.outputSize, self.outputSize)(ZFOI) -- batchSize x inputSize
  if numGates >= 3 then
    o = nn.Narrow(2, 1 + 2 * self.outputSize, self.outputSize)(ZFOI) -- batchSize x inputSize
  end
  if numGates >= 4 then
    i = nn.Narrow(2, 1 + 3 * self.outputSize, self.outputSize)(ZFOI) -- batchSize x inputSize
  end

--  if numGates == 2 then
--    prevH = nn.Identity()() 
--    table.insert(inputs, prevH)
--  end
--
--  if numGates >= 3 then
--    prevC = nn.Identity()() -- batchSize x inputSize
--    table.insert(inputs, prevC)
--  end


  local h_t, c_t

  -- Pooling
--  if self.poolingMethod == 'f' then
--    i = nn.AddConstant(1, true)(nn.MulConstant(-1, true)(f))
--    h_t = nn.CAddTable()({
--      nn.CMulTable()({f, prevH}),
--      nn.CMulTable()({i, z})
--    })
--  else
  if self.poolingMethod == 'fo' then
    i = nn.AddConstant(1, true)(nn.MulConstant(-1, true)(f))
    c_t = nn.CAddTable()({
      nn.CMulTable()({f, prevC}),
      nn.CMulTable()({i, z})
    })
    h_t = nn.CMulTable()({o, c_t})
  elseif self.poolingMethod == 'ifo' then
    c_t = nn.CAddTable()({
      nn.CMulTable()({f, prevC}),
      nn.CMulTable()({i, z})
    })
    h_t = nn.CMulTable()({o, c_t})
  else
    print('QRNNCNN: Invalid poolingMethod: ' .. self.poolingMethod)
    os.exit()
  end

  return nn.Sequential()
            :add(
              nn.ConcatTable():add(nn.SelectTable(1))
                              :add(nn.SelectTable(3))
            )
            :add(nn.gModule({ZFOI, prevC}, {h_t, c_t}))
end

function TestQRNNRNN:getHiddenState(step, input)
   step = step == nil and (self.step - 1) or (step < 0) and (self.step - step - 1) or step
   local prevOutput, prevCell
   if step == 0 then
      prevOutput = self.userPrevOutput or self.outputs[step] or self.zeroTensor
      prevCell = self.userPrevCell or self.cells[step] or self.zeroTensor
      if input then
         if input:dim() == 2 then
            self.zeroTensor:resize(input:size(1), self.outputSize):zero()
         else
            self.zeroTensor:resize(self.outputSize):zero()
         end
      end
   else
      -- previous output and cell of this module
      prevOutput = self.outputs[step]
      prevCell = self.cells[step]
   end
   return {prevOutput, prevCell}
end

function TestQRNNRNN:setHiddenState(step, hiddenState)
   step = step == nil and (self.step - 1) or (step < 0) and (self.step - step - 1) or step
   assert(torch.type(hiddenState) == 'table')
   assert(#hiddenState == 2)

   -- previous output of this module
   self.outputs[step] = hiddenState[1]
   self.cells[step] = hiddenState[2]
end

------------------------- forward backward -----------------------------
function TestQRNNRNN:updateOutput(input)
   local prevOutput, prevCell = unpack(self:getHiddenState(self.step-1, input))

   -- output(t), cell(t) = TestQRNNRNN{input(t), output(t-1), cell(t-1)}
   local output, cell
   if self.train ~= false then
      self:recycle()
      local recurrentModule = self:getStepModule(self.step)
      -- the actual forward propagation
      print('input at step ' .. self.step)
      print(input)
      output, cell = unpack(recurrentModule:updateOutput{input, prevOutput, prevCell})
   else
      output, cell = unpack(self.recurrentModule:updateOutput{input, prevOutput, prevCell})
   end

   self.outputs[self.step] = output
   self.cells[self.step] = cell

   self.output = output
   self.cell = cell

   self.step = self.step + 1
   self.gradPrevOutput = nil
   self.updateGradInputStep = nil
   self.accGradParametersStep = nil
   -- note that we don't return the cell, just the output
   return self.output
end

function TestQRNNRNN:getGradHiddenState(step)
   self.gradOutputs = self.gradOutputs or {}
   self.gradCells = self.gradCells or {}
   local _step = self.updateGradInputStep or self.step
   step = step == nil and (_step - 1) or (step < 0) and (_step - step - 1) or step
   local gradOutput, gradCell
   if step == self.step-1 then
      gradOutput = self.userNextGradOutput or self.gradOutputs[step] or self.zeroTensor
      gradCell = self.userNextGradCell or self.gradCells[step] or self.zeroTensor
   else
      gradOutput = self.gradOutputs[step]
      gradCell = self.gradCells[step]
   end
   return {gradOutput, gradCell}
end

function TestQRNNRNN:setGradHiddenState(step, gradHiddenState)
   local _step = self.updateGradInputStep or self.step
   step = step == nil and (_step - 1) or (step < 0) and (_step - step - 1) or step
   assert(torch.type(gradHiddenState) == 'table')
   assert(#gradHiddenState == 2)

   self.gradOutputs[step] = gradHiddenState[1]
   self.gradCells[step] = gradHiddenState[2]
end

function TestQRNNRNN:_updateGradInput(input, gradOutput)
   assert(self.step > 1, "expecting at least one updateOutput")
   local step = self.updateGradInputStep - 1
   assert(step >= 1)

   -- set the output/gradOutput states of current Module
   local recurrentModule = self:getStepModule(step)

   -- backward propagate through this step
   local gradHiddenState = self:getGradHiddenState(step)
   local _gradOutput, gradCell = gradHiddenState[1], gradHiddenState[2]
   assert(_gradOutput and gradCell)

   self._gradOutputs[step] = nn.rnn.recursiveCopy(self._gradOutputs[step], _gradOutput)
   nn.rnn.recursiveAdd(self._gradOutputs[step], gradOutput)
   gradOutput = self._gradOutputs[step]

   local inputTable = self:getHiddenState(step-1)
   table.insert(inputTable, 1, input)

   local gradInputTable = recurrentModule:updateGradInput(inputTable, {gradOutput, gradCell})

   local _ = require 'moses'
   self:setGradHiddenState(step-1, _.slice(gradInputTable, 2, 3))

   return gradInputTable[1]
end

function TestQRNNRNN:_accGradParameters(input, gradOutput, scale)
   local step = self.accGradParametersStep - 1
   assert(step >= 1)

   -- set the output/gradOutput states of current Module
   local recurrentModule = self:getStepModule(step)

   -- backward propagate through this step
   local inputTable = self:getHiddenState(step-1)
   table.insert(inputTable, 1, input)
   local gradOutputTable = self:getGradHiddenState(step)
   gradOutputTable[1] = self._gradOutputs[step] or gradOutputTable[1]
   recurrentModule:accGradParameters(inputTable, gradOutputTable, scale)
end

function TestQRNNRNN:clearState()
   self.zeroTensor:set()
   if self.userPrevOutput then self.userPrevOutput:set() end
   if self.userPrevCell then self.userPrevCell:set() end
   if self.userGradPrevOutput then self.userGradPrevOutput:set() end
   if self.userGradPrevCell then self.userGradPrevCell:set() end
   return parent.clearState(self)
end

function TestQRNNRNN:type(type, ...)
   if type then
      self:forget()
      self:clearState()
      self.zeroTensor = self.zeroTensor:type(type)
   end
   return parent.type(self, type, ...)
end