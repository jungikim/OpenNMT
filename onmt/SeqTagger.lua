--[[ Sequence tagger. ]]
local SeqTagger, parent = torch.class('SeqTagger', 'Model')

local options = {
  {
    '-word_vec_size', { 500 },
    [[List of embedding sizes: `word[ feat1[ feat2[ ...] ] ]`.]],
    {
      structural = 0
    }
  },
  {
    '-pre_word_vecs_enc', '',
    [[Path to pretrained word embeddings on the encoder side serialized as a Torch tensor.]],
    {
      valid = onmt.utils.ExtendedCmdLine.fileNullOrExists,
      init_only = true
    }
  },
  {
    '-fix_word_vecs_enc', false,
    [[Fix word embeddings on the encoder side.]],
    {
      enum = { false, true, 'pretrained' },
      structural = 1
    }
  },
  {
    '-feat_merge', 'concat',
    [[Merge action for the features embeddings.]],
    {
      enum = {'concat', 'sum'},
      structural = 0
    }
  },
  {
    '-feat_vec_exponent', 0.7,
    [[When features embedding sizes are not set and using `-feat_merge concat`, their dimension
      will be set to `N^feat_vec_exponent` where `N` is the number of values the feature takes.]],
    {
      structural = 0
    }
  },
  {
    '-feat_vec_size', 20,
    [[When features embedding sizes are not set and using `-feat_merge sum`,
      this is the common embedding size of the features]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt(),
      structural = 0
    }
  },
  {
    '-loglikelihood', 'word',
    [[Specifies the type of loglikelihood of the tagger model;
    'word' indicates tags are predicted at the word-level, and
    'sentence' indicates tagging process is treated as a markov chain]],
    {
      enum = {'word', 'sentence', 'ctc'},
      structural = 1
    }
  },
  {
    '-weopt_pretrained', '',
    [[Specifies the pre-trained word embedding that the network should try to optimize]],
    {
      structural = 1
    }
  },
  {
    '-weopt_dict', '',
    [[Specifies the dictionary for the pre-trained word embedding that the network should try to optimize]],
    {
      structural = 1
    }
  },
  {
    '-weopt_loss', 'mse',
    [[Specifies the loss function for the pre-trained word embedding optimization]],
    {
      structural = 1
    }
  },
  {
    '-weopt_model', 'lstm',
    [[Specifies the model that learns the pre-trained word embedding from underlying model]],
    {
      structural = 1
    }
  },
  {
    '-weopt_ctc_char_concat', false,
    [[Specifies whether or not the output of ctc is char and should be concatenated to form words]],
    {
      structural = 1
    }
  }
}


function SeqTagger.declareOpts(cmd)
  cmd:setCmdLineOptions(options, SeqTagger.modelName())
  onmt.Encoder.declareOpts(cmd)
  onmt.Factory.declareOpts(cmd)
end

function SeqTagger:__init(args, dicts)
  parent.__init(self, args)
  onmt.utils.Table.merge(self.args, onmt.utils.ExtendedCmdLine.getModuleOpts(args, options))

  if not dicts.src then
    -- the input is already a vector
    args.dimInputSize = dicts.srcInputSize
  end

  self.models.encoder = onmt.Factory.buildWordEncoder(args, dicts.src)
  self.models.generator = onmt.Factory.buildGenerator(args, dicts.tgt)

  onmt.utils.Error.assert(args.loglikelihood == 'word' or args.loglikelihood == 'sentence' or args.loglikelihood == 'ctc',
    'Invalid loglikelihood type of SeqTagger `%s\'', args.loglikelihood)

  self.loglikelihood = args.loglikelihood
  _G.logger:info('SeqTagger loglikelihood: ' .. self.loglikelihood)

  if self.loglikelihood == 'word' then
    self.criterion = onmt.ParallelClassNLLCriterion(onmt.Factory.getOutputSizes(dicts.tgt))
  elseif self.loglikelihood == 'sentence' then
    self.criterion = onmt.SentenceNLLCriterion(args, onmt.Factory.getOutputSizes(dicts.tgt))
    self.models.criterion = self.criterion -- criterion is model parameter
  elseif self.loglikelihood == 'ctc' then
    require 'nnx'
    self.criterion = nn.CTCCriterion(--[[batchFirst]] true) -- with batchFirst, expects B x seqLen x dim as input
    self:loadCtcDecoder(dicts)
  end

  if args.weopt_pretrained and args.weopt_pretrained:len() > 0 then

    self.weopt_pretrained = torch.load(args.weopt_pretrained)

    print('self.weopt_pretrained:size(1):'.. self.weopt_pretrained:size(1))
    print('self.weopt_pretrained:size(2):'.. self.weopt_pretrained:size(2))

    self.weopt_dict = onmt.utils.Dict.new(args.weopt_dict)

    if args.weopt_model == 'lstm' then
      self.models.weopt = onmt.LSTM.new(--[[layers]] 1,
                                        --[[inputSize]] self.models.encoder.args.rnnSize,
                                        --[[hiddenSize]] self.weopt_pretrained:size(2),
                                        --[[dropout]] 0,
                                        --[[residual]] false,
                                        --[[dropout_input]] false,
                                        --[[dropout_type]] '')
--    elseif args.weopt_model == 'cnn' then
    else
      _G.logger:error('Model option weopt_model=' .. args.weopt_model .. ' is invalid or not implemented yet')
      os.exit(1)
    end

    self.weopt_criterion = nn.MSECriterion()
    self.weopt_criterion.siveAverage = false
  end

  if args and
     args.preprocess and
     args.preprocess.audio_feature_type then
    self.audio_feature_type = args.preprocess.audio_feature_type
  end
end

function SeqTagger:loadCtcDecoder(dicts, ctc_nbest, ctc_lm, ctc_lm_weight)
  local ctcdecode = require 'ctcdecode'
  local paths = require 'paths'

  self.ctc_nBest =  ctc_nbest or 3
  self.ctc_lm_weight = ctc_lm_weight or 1.0

  if ctc_lm and ctc_lm:len() > 0 then
    if not paths.filep(ctc_lm) then
      _G.logger:error('Cannot find CTC LM file: ' .. ctc_lm)
      os.exit(1)
    end

    local tmpF = paths.tmpname()
    _G.logger:info('Writing temporary label file to ' .. tmpF)
    local file = assert(io.open(tmpF, 'w'))
    for i = 1, dicts.tgt.words:size() do
      file:write(dicts.tgt.words.idxToLabel[i] .. '\n')
    end
    file:close()

    local labelMapFilename = paths.thisfile(tmpF)
    local ngramModelFilename = paths.thisfile(ctc_lm)

    local scorer = ctcdecode.NGramBeamScorer(labelMapFilename, ngramModelFilename)
--    CTC beam search decoder as described in https://arxiv.org/abs/1408.2873
--    scorer:setNGramModelWeight(float) 1.0?
--    scorer:setWordInsertionWeight(float) 0.0?

    scorer:setNGramModelWeight(self.ctc_lm_weight)

    self.ctc_decoder = ctcdecode.NGramDecoder(
      --[[numClasses]]dicts.tgt.words:size(),
      --[[beamWidth]]10 * self.ctc_nBest,
      --[[scorer]]scorer,
      --[[blankLabel]]1,
      --[[mergeRepeated]]false
      )
--    https://research.google.com/pubs/pub44823.html
--    Default is to do no label selection
--    self.ctc_decoder:SetLabelSelectionSize(int) 40
      -- how many items in each beam are passed through to the beam scorer.
      -- Only items with top N input scores are considered.
--    self.ctc_decoder:SetLabelSelectionMargin(float) 1.0
      -- controls the difference between minimal input score (versus the best scoring label) for an item to be passed to the beam scorer.
      -- This margin is expressed in terms of log-probability.
      -- -1 for unlimited;
  else
    local scorer = ctcdecode.DefaultBeamScorer()

    self.ctc_decoder = ctcdecode.BeamSearchDecoder(
      --[[numClasses]]dicts.tgt.words:size(),
      --[[beamWidth]]10 * self.ctc_nBest,
      --[[scorer]]scorer,
      --[[blankLabel]]1,
      --[[mergeRepeated]]false
    )
  end
end

function SeqTagger.load(args, models, dicts)
  local self = torch.factory('SeqTagger')()

  parent.__init(self, args)
  onmt.utils.Table.merge(self.args, onmt.utils.ExtendedCmdLine.getModuleOpts(args, options))

  self.models.encoder = onmt.Factory.loadEncoder(models.encoder)
  self.models.generator = onmt.Factory.loadGenerator(models.generator)

  onmt.utils.Error.assert(args.loglikelihood == 'word' or args.loglikelihood == 'sentence' or args.loglikelihood == 'ctc',
                                  'Invalid loglikelihood type of SeqTagger `%s\'', args.loglikelihood)

  if args.loglikelihood == 'word' then
    self.criterion = onmt.ParallelClassNLLCriterion(onmt.Factory.getOutputSizes(dicts.tgt))
    self.loglikelihood = 'word'
  elseif args.loglikelihood == 'sentence' then
    if not models.criterion then -- loading pre-trained word model to further train sentence model
      _G.logger:info('Creating a new SentenceNLLCriterion')
      self.criterion = onmt.SentenceNLLCriterion(args, onmt.Factory.getOutputSizes(dicts.tgt))
      local p, g = self.criterion:getParameters()
      p:uniform(-args.param_init, args.param_init)
      g:uniform(-args.param_init, args.param_init)
      self.criterion:postParametersInitialization()
    else
      self.criterion = onmt.Factory.loadSentenceNLLCriterion(models.criterion)
    end
    self.models.criterion = self.criterion
    self.loglikelihood = 'sentence'
  elseif args.loglikelihood == 'ctc' then
    require 'nnx'
    self.criterion = nn.CTCCriterion(--[[batchFirst]] true) -- with batchFirst, expects B x seqLen x dim as input
    self.loglikelihood = 'ctc'
    self:loadCtcDecoder(dicts)
  end

  if args.weopt_pretrained and args.weopt_pretrained:len() > 0 then

    self.weopt_pretrained = torch.load(args.weopt_pretrained)

    print('self.weopt_pretrained:size(1):'.. self.weopt_pretrained:size(1)) -- vocab size
    print('self.weopt_pretrained:size(2):'.. self.weopt_pretrained:size(2)) -- embedding size

    self.weopt_dict = onmt.utils.Dict.new(args.weopt_dict)

    if not args.weopt_model then
      if args.weopt_model == 'lstm' then
        self.models.weopt = onmt.LSTM.new(--[[layers]] 1,
                                          --[[inputSize]] self.models.encoder.args.rnnSize,
                                          --[[hiddenSize]] self.weopt_pretrained:size(2),
                                          --[[dropout]] 0,
                                          --[[residual]] false,
                                          --[[dropout_input]] false,
                                          --[[dropout_type]] '')
  --    elseif args.weopt_model == 'cnn' then
      else
        _G.logger:error('Model option weopt_model=' .. args.weopt_model .. ' is invalid or not implemented yet')
        os.exit(1)
      end
    else
      _G.logger:info('Model has WEOPT.')
      if self.models.weopt.outputSize ~= self.weopt_pretrained:size(2) then
        _G.logger:error('Output size of WEOPT model ' .. self.models.weopt.outputSize .. ' does not match pretrained embedding size ' .. self.weopt_pretrained:size(2))
        os.exit(1)
      end
    end

    self.weopt_criterion = nn.MSECriterion()
    self.weopt_criterion.siveAverage = false
  end

  if args and
     args.preprocess and
     args.preprocess.audio_feature_type then
    self.audio_feature_type = args.preprocess.audio_feature_type
    _G.logger:info('Model is trained with audio_feature_type: ' .. self.audio_feature_type)
  end

  return self
end

-- Returns model name.
function SeqTagger.modelName()
  return 'Sequence Tagger'
end

-- Returns expected default datatype or if passed a parameter, returns if it is supported
function SeqTagger.dataType(datatype)
  if not datatype then
    return 'bitext'
  else
    return datatype == 'bitext' or datatype == 'feattext' or datatype == 'audiotext'
  end
end

function SeqTagger:enableProfiling()
  _G.profiler.addHook(self.models.encoder, 'encoder')
  _G.profiler.addHook(self.models.generator, 'generator')
  _G.profiler.addHook(self.criterion, 'criterion')
end

function SeqTagger:getOutput(batch)
  return batch.targetOutput
end

function SeqTagger:forwardComputeLoss(batch)
  local _, context = self.models.encoder:forward(batch) -- B x SeqLen x dim

  local loss = 0

  if self.loglikelihood == 'sentence' then
    local reference = batch.targetOutput:t() -- SeqLen x B  -> B x SeqLen
    local tagsScoreTable = {}
    for t = 1, batch.sourceLength do
      local tagsScore = self.models.generator:forward(context:select(2, t)) -- B x TagSize
      -- tagsScore is a table
      tagsScore = nn.utils.addSingletonDimension(tagsScore[1], 3):clone() -- B x TagSize x 1
      table.insert(tagsScoreTable, tagsScore)
    end
    local tagsScores = nn.JoinTable(3):forward(tagsScoreTable)  -- B x TagSize x SeqLen
    loss = self.models.criterion:forward(tagsScores, reference)

  elseif self.loglikelihood == 'ctc' then

    local tagsScoreTable = {}
    for t = 1, context:size(2) do
      local tagsScore = self.models.generator:forward(context:select(2, t)) -- B x TagSize
      -- tagsScore is a table
      tagsScore = nn.utils.addSingletonDimension(tagsScore[1], 2):clone() -- B x 1 x TagSize
      table.insert(tagsScoreTable, tagsScore)
    end
    local tagsScores = nn.JoinTable(2):forward(tagsScoreTable):type(torch.type(context)) -- B x SeqLen x TagSize

    local reference = batch.targetOutput:t() -- SeqLen x B -> B x SeqLen

    local refTable = {}
    for bIdx = 1, batch.size do
      table.insert(refTable,torch.totable(reference[{{bIdx},{1, batch.targetSize[bIdx]-1}}][1]:clone():float()))
    end

    loss = self.criterion:forward(tagsScores, refTable, batch.sourceSize:float())

  else -- 'word'
    for t = 1, batch.sourceLength do
      local genOutputs = self.models.generator:forward(context:select(2, t))

      local output = batch:getTargetOutput(t)

      -- Same format with and without features.
      if torch.type(output) ~= 'table' then output = { output } end

      loss = loss + self.criterion:forward(genOutputs, output)
    end
  end

  return loss
end

function SeqTagger:trainNetwork(batch)
  local loss = 0

  local _, context = self.models.encoder:forward(batch)

  local gradContexts = context:clone():zero()

  if self.loglikelihood == 'sentence' then

    local reference = batch.targetOutput:t() -- SeqLen x B  -> B x SeqLen
    local B = batch.size
    local T = batch.sourceLength

    local tagsScoreTable = {}
    for t = 1, T do
      local tagsScore = self.models.generator:forward(context:select(2, t)) -- B x TagSize
      -- tagsScore is a table
      tagsScore = nn.utils.addSingletonDimension(tagsScore[1], 3):clone() -- B x TagSize x 1
      table.insert(tagsScoreTable, tagsScore)
    end
    local tagsScores = nn.JoinTable(3):forward(tagsScoreTable) -- B x TagSize x SeqLen

    loss = loss + self.criterion:forward(tagsScores, reference)

    local gradCriterion = self.models.criterion:backward(tagsScores, reference) -- B x TagSize x SeqLen

    gradCriterion = torch.div(gradCriterion, B)
    for t = 1, T do
      gradContexts:select(2,t):copy(self.models.generator:backward(context:select(2, t), {gradCriterion:select(3,t)}))
    end

  elseif self.loglikelihood == 'ctc' then

    local tagsScoreTable = {}
    for t = 1, context:size(2) do
      local tagsScore = self.models.generator:forward(context:select(2, t)) -- B x TagSize
      -- tagsScore is a table
      tagsScore = nn.utils.addSingletonDimension(tagsScore[1], 2):clone() -- B x 1 x TagSize
      table.insert(tagsScoreTable, tagsScore)
    end
    local tagsScores = nn.JoinTable(2):forward(tagsScoreTable):type(torch.type(context)) -- B x SeqLen x TagSize

    local reference = batch.targetOutput:t() -- SeqLen x B -> B x SeqLen

    local refTable = {}
    for bIdx = 1, batch.size do
      table.insert(refTable,torch.totable(reference[{{bIdx},{1, batch.targetSize[bIdx]-1}}][1]:clone():float()))
    end

--        print('tagsScores: ' .. tostring(tagsScores))
--        print('refTable: ')
--        for bIdx = 1, batch.size do
--          print(refTable[bIdx])
--        end
--        print('batch.sourceSize: ' .. tostring(batch.sourceSize))

    loss = self.criterion:forward(tagsScores, refTable, batch.sourceSize:float())

--        print('Loss: ' .. tostring(loss))

    local gradCriterion = self.criterion:backward(tagsScores, refTable)

    if loss ~= loss then
      print('Invalid loss: ' .. tostring(loss))
      gradCriterion:zero()
      loss = 0.0
    end

    gradCriterion = torch.div(gradCriterion, batch.size)
    for t = 1, context:size(2) do
      gradContexts:select(2,t):copy(self.models.generator:backward(context:select(2, t), {gradCriterion:select(2,t)}))
    end



    if self.models.weopt then

      for b = 1, batch.size do
        -- expects tagsScores to be seqLen x B x tagSize
        local outputs, alignments, pathLen, scores =
                     self.ctc_decoder:decode(tagsScores[{{b},{batch.sourceLength - batch.sourceSize[b] + 1, batch.sourceLength},{}}]:transpose(1,2):type('torch.FloatTensor'),
                                             self.ctc_nBest,
                                             torch.IntTensor({batch.sourceSize[b]}))

        print('Outputs: ')
        print(outputs)

        print('alignments: ')
        print(alignments)

        os.exit(1)

--        for t = 1, pathLen[1][--[[Best]] 1] do
--          pred[b][t] = outputs[1][--[[Best]] 1][t]
--          feats[b][t] = {}
--        end
      end


    end



  else -- 'word'
    -- For each word of the sentence, generate target.
    for t = 1, batch.sourceLength do
      local genOutputs = self.models.generator:forward(context:select(2, t))

      local output = batch:getTargetOutput(t)

      -- Same format with and without features.
      if torch.type(output) ~= 'table' then output = { output } end

      loss = loss + self.criterion:forward(genOutputs, output)
      local genGradOutput = self.criterion:backward(genOutputs, output)

      for j = 1, #genGradOutput do
        genGradOutput[j]:div(batch.totalSize)
      end

      gradContexts[{{}, t}]:copy(self.models.generator:backward(context:select(2, t), genGradOutput))
    end
  end

  self.models.encoder:backward(batch, nil, gradContexts)

  return loss
end

function SeqTagger:tagBatch(batch)
  local pred = {}
  local feats = {}

  for _ = 1, batch.size do
    table.insert(pred, {})
    table.insert(feats, {})
  end
  local _, context = self.models.encoder:forward(batch)

  if self.loglikelihood == 'sentence' then

    local tagsScoreTable = {}
    for t = 1, batch.sourceLength do
      local tagsScore = self.models.generator:forward(context:select(2, t)) -- B x TagSize
      -- tagsScore is a table
      tagsScore = nn.utils.addSingletonDimension(tagsScore[1], 3):clone() -- B x TagSize x 1
      table.insert(tagsScoreTable, tagsScore)
    end
    local tagsScores = onmt.utils.Cuda.convert(nn.JoinTable(3):forward(tagsScoreTable))  -- B x TagSize x SeqLen

    -- viterbi search
    local senPreds = self.criterion:viterbiSearch(tagsScores, batch.sourceSize) -- B x SeqLen (type Long)

    for t = 1, batch.sourceLength do
      for b = 1, batch.size do
        -- padded in the beginning
        if t > batch.sourceLength - batch.sourceSize[b] then
          pred[b][t - batch.sourceLength + batch.sourceSize[b]] = senPreds[b][t]
          feats[b][t - batch.sourceLength + batch.sourceSize[b]] = {}
        end
      end
    end

  elseif self.loglikelihood == 'ctc' then

    local tagsScoreTable = {}
    for t = 1, batch.sourceLength do
      local tagsScore = self.models.generator:forward(context:select(2, t)) -- B x TagSize
      -- tagsScore is a table
      tagsScore = nn.utils.addSingletonDimension(tagsScore[1], 2):clone() -- B x 1 x TagSize
      table.insert(tagsScoreTable, tagsScore)
    end
    local tagsScores = nn.JoinTable(2):forward(tagsScoreTable):type(torch.type(context)) -- B x SeqLen x TagSize

    --    print('tagsScores: ' .. tostring(tagsScores))
    --    print('batch.sourceSize: ' .. tostring(batch.sourceSize))

    for b = 1, batch.size do
      -- expects tagsScores to be seqLen x B x tagSize
      local outputs, --[[alignments]]_, pathLen, --[[scores]]_ =
                                self.ctc_decoder:decode(tagsScores[{{b},{batch.sourceLength - batch.sourceSize[b] + 1, batch.sourceLength},{}}]:transpose(1,2):type('torch.FloatTensor'),
                                                    self.ctc_nBest,
                                                    torch.IntTensor({batch.sourceSize[b]}))
      for t = 1, pathLen[1][--[[Best]] 1] do
        pred[b][t] = outputs[1][--[[Best]] 1][t]
        feats[b][t] = {}
      end
    end

-- TODO: batched decoding gives different result due to left-padded source?
--    local outputs, alignments, pathLen, scores = decoder:decode(tagsScores:transpose(1,2), nBest, batch.sourceSize:type('torch.IntTensor'))
--    for b = 1, batch.size do
--      for t = 1, pathLen[b][--[[Best]] 1] do
--        pred[b][t] = outputs[b][--[[Best]] 1][t]
--        feats[b][t] = {}
--      end
--    end

  else -- 'word'
    for t = 1, batch.sourceLength do
      local out = self.models.generator:forward(context:select(2, t))
      if type(out[1]) == 'table' then
        out = out[1]
      end
      local _, best = out[1]:max(2)
      for b = 1, batch.size do
        if t > batch.sourceLength - batch.sourceSize[b] then
          pred[b][t - batch.sourceLength + batch.sourceSize[b]] = best[b][1]
          feats[b][t - batch.sourceLength + batch.sourceSize[b]] = {}
        end
      end
      for j = 2, #out do
        _, best = out[j]:max(2)
        for b = 1, batch.size do
          if t > batch.sourceLength - batch.sourceSize[b] then
            feats[b][t - batch.sourceLength + batch.sourceSize[b]][j - 1] = best[b][1]
          end
        end
      end
    end

  end

  return pred, feats
end

return SeqTagger
