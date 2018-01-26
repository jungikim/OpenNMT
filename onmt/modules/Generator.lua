--[[ Default decoder generator.
     Given RNN state, produce categorical distribution for tokens and features

     Simply implements $$softmax(W h b)$$.

     version 2: merge FeaturesGenerator and Generator - the generator nn is a table
--]]
local Generator, parent = torch.class('onmt.Generator', 'onmt.Network')

-- for back compatibility - still declare FeaturesGenerator - but no need to define it
torch.class('onmt.FeaturesGenerator', 'onmt.Generator')

function Generator:__init(opt, sizes)
  parent.__init(self)
  self:_buildGenerator(opt, sizes)
  -- for backward compatibility with previous model
  self.version = 2
end

function Generator:_buildGenerator(opt, sizes)
  local generator = nn.ConcatTable()
  local rnn_size = opt.rnn_size

  for i = 1, #sizes do
    local linear = nn.Linear(rnn_size, sizes[i])
    if i == 1 then
      self.rindexLinear = linear
    end
    generator:add(nn.Sequential()
                    :add(linear)
                    :add(nn.LogSoftMax()))
  end

  self:set(generator)
end

--[[ If the target vocabulary for the batch is not full vocabulary ]]
function Generator:setGeneratorVocab(t)
  self.rindexLinear:RIndex_setOutputIndices(t)
end

--[[ Release Generator for inference only ]]
function Generator:release()
  if self.rindexLinear then
    self.rindexLinear:RIndex_clean()
  end
end

function Generator.load(generator)
  -- Ensure backward compatibility with previous generators.

  if not generator.version then
    if torch.typename(generator) == 'onmt.Generator' then
      generator:set(nn.ConcatTable():add(generator.net))
    elseif torch.typename(generator) == 'onmt.FeaturesGenerator' then
      if torch.typename(generator.net.modules[1]:get(1)) == 'onmt.Generator' then
        generator.net.modules[1] = generator.net.modules[1]:get(1).net
      end
    end
    generator.version = 2
  end

  if not generator.rindexLinear then
    generator:apply(function(m)
      if torch.typename(m) == 'nn.Linear' then
        if not generator.rindexLinear then
          generator.rindexLinear = m
        end
      end
    end)
  end

  return generator
end

function Generator:applySVDSoftmax()
  -- replace nn.Linear with nn.SVDLinear
  local svdReplaced = false
  self:replace(function(module)
    -- assume the first nn.Linear we find is for the main feature (token)
    if torch.typename(module) == 'nn.Linear' and not svdReplaced then
      local mod = nn.SVDLinear(module.weight:size(2), module.weight:size(1), module.bias ~= nil)
      mod:load(module)
      mod:doSVD()
      mod:type(module._type)
      _G.logger:info('nn.Linear ' .. tostring(module) .. ' replaced with ' .. tostring(mod))
      svdReplaced = true
      return mod
    else
      return module
    end
  end
  )
end

function Generator:applySVDParam(W, N)
  self:replace(
    function(module)
      if torch.typename(module) == 'nn.SVDLinear' then
        module:setSVDParam(W, N)
        _G.logger:info('Setting W, N of nn.SVDLinear ' .. tostring(module) .. ' to ' .. tostring(W) .. ' and ' .. tostring(N))
      end
      return module
    end
  )
end

function Generator:stopSVDSoftmax()
  -- replace nn.SVDLinear with nn.Linear
  self:replace(function(module)
    if torch.typename(module) == 'nn.SVDLinear' then
      local mod = nn.Linear(module.weight:size(2), module.weight:size(1), module.bias ~= nil)
      mod.weight = module.weight
      mod.bias = module.bias
      mod:type(module._type)
      _G.logger:info('nn.SVDLinear ' .. tostring(module) .. ' replaced with ' .. tostring(mod))
      return mod
    else
      return module
    end
  end
  )
end

function Generator:updateOutput(input)
  input = type(input) == 'table' and input[1] or input
  self.output = self.net:updateOutput(input)
  return self.output
end

function Generator:updateGradInput(input, gradOutput)
  input = type(input) == 'table' and input[1] or input
  self.gradInput = self.net:updateGradInput(input, gradOutput)
  return self.gradInput
end

function Generator:accGradParameters(input, gradOutput, scale)
  input = type(input) == 'table' and input[1] or input
  self.net:accGradParameters(input, gradOutput, scale)
end
