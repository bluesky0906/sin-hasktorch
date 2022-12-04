{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE GADTs #-}


module SinRNN where
import Util
import Torch
import Torch.Layer.RNN (RNNHypParams(..), RNNParams, rnnLayer)
import Torch.Layer.Linear
import Torch.Control (mapAccumM)
import Graphics.Gnuplot.Simple
import GHC.Generics


step_per_cycle = 100
number_of_cycles = 1


data SinRNNSpec = SinRNNSpec {
  modelDevice :: Device,
  hiddenSize :: Int
} deriving (Show, Eq)

data SinRNN where
  SinRNN ::
    {
      rnn :: RNNParams,
      h0 :: Parameter,
      linear :: LinearParams
    } ->
    SinRNN
  deriving (Show, Generic, Parameterized)

instance
  Randomizable
    SinRNNSpec
    SinRNN
  where
    sample SinRNNSpec {..} =
      SinRNN
        <$> sample (RNNHypParams modelDevice 1 hiddenSize)
        <*> (makeIndependent =<< randnIO' [hiddenSize])
        <*> sample (LinearHypParams modelDevice hiddenSize 1)

sinRNN ::
  SinRNN ->
  [Tensor] ->
  [Tensor]
sinRNN SinRNN {..} input =
  let (output, _) = rnnLayer rnn (toDependent $ h0) input
  in fmap (linearLayer linear) output

main :: IO()
main = do
  let times = [0..(step_per_cycle * number_of_cycles - 1)]
      sin_t = fmap (sin_wave step_per_cycle) times
      sin_t1 = tail sin_t ++ [sin_wave step_per_cycle (step_per_cycle * number_of_cycles)]
      -- テストデータ
      cos_t = fmap (cos_wave step_per_cycle) times
      cos_t1 = tail cos_t ++ [cos_wave step_per_cycle (step_per_cycle * number_of_cycles)]

  initModel <- sample $ SinRNNSpec (Device CPU 0) 100
  -- | setting
  let numEpochs = 100
      optim = mkAdam 1 0.9 0.999 (flattenParameters initModel)
      -- optim = GD
  plotWave "sin" (times, sin_t) (times, sin_t1)

  ((trainedModel, _), losses) <- mapAccumM [1..numEpochs] (initModel, optim) $ \epoc (model, opt) -> do
    let x_tensor = fmap ((toDType Float) . reshape [1] . asTensor) sin_t
        t_tensor = fmap ((toDType Float) . reshape [1] . asTensor) sin_t1
        prediction = fmap (reshape [1]) (sinRNN model x_tensor)
        loss = mseLoss (cat (Dim 0) prediction) (cat (Dim 0) t_tensor)
    updated <- runStep model opt loss 1e-2
    print $ "Training Loss: " ++ (show loss)
    return (updated, loss)

  let x_tensor = fmap ((toDType Float) . reshape [1] . asTensor) sin_t
      pred = fmap toDouble $ sinRNN trainedModel x_tensor
  plotWave "untyped-sinRNN_pred" (times, sin_t1) (times, pred)

  let x_tensor = fmap ((toDType Float) . reshape [1] . asTensor) cos_t
      pred = fmap toDouble $ sinRNN trainedModel x_tensor
  plotWave "untyped-sinRNN_eval" (times, cos_t1) (times, pred)
  return ()
