{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}


module SinRNN where
import RNN
import Util
import Torch.Typed
import qualified Torch.Tensor as D
import Torch.Control (mapAccumM)
import Graphics.Gnuplot.Simple
import GHC.Generics
import GHC.TypeNats
import System.Process


step_per_cycle = 100
number_of_cycles = 1

type ModelDevice = '( 'CPU, 0)
type HiddenSize = 2
type Dtype = 'Double

data SinRNNSpec
  = SinRNNSpec
  deriving (Show, Eq)

data SinRNN where
  SinRNN ::
    {
      rnn :: RNN 1 HiddenSize Dtype ModelDevice,
      linear :: Linear HiddenSize 1 Dtype ModelDevice
    } ->
    SinRNN
  deriving (Show, Generic, Parameterized)

instance
  Randomizable
    SinRNNSpec
    SinRNN
  where
    sample SinRNNSpec =
      SinRNN
        <$> sample (RNNSpec @1 @HiddenSize @Dtype @ModelDevice)
        <*> sample (LinearSpec @HiddenSize @1 @Dtype @ModelDevice)

sinRNN ::
  SinRNN ->
  [Tensor ModelDevice Dtype '[1]] ->
  [Tensor ModelDevice Dtype '[1]]
sinRNN SinRNN {..} input =
  fmap (linearForward linear) (rnnForward rnn input)


meanSquaredError :: 
  [Tensor ModelDevice Dtype '[1]] ->
  [Tensor ModelDevice Dtype '[1]] ->
  Tensor ModelDevice Dtype '[]
meanSquaredError y t = (select @0 @0 squaredSum) / (UnsafeMkTensor @ModelDevice @Dtype @'[] $ D.asTensor (Prelude.length y))
  where
    squared = fmap (\(a, b) -> (a - b) ^ 2) (zip y t)
    squaredSum = foldr1 (+) squared

main :: IO()
main = do
  let times = [0..(step_per_cycle * number_of_cycles - 1)]
      sin_t = fmap (sin_wave step_per_cycle) times
      sin_t1 = tail sin_t ++ [sin_wave step_per_cycle (step_per_cycle * number_of_cycles)]
      -- テストデータ
      cos_t = fmap (cos_wave step_per_cycle) times
      cos_t1 = tail cos_t ++ [cos_wave step_per_cycle (step_per_cycle * number_of_cycles)]

  initModel <- sample SinRNNSpec
  -- | setting
  let numEpochs = 100
      optim = mkAdam 0.001 0.9 0.999 (flattenParameters initModel)
      -- optim = GD
  plotWave "sin" (times, sin_t) (times, sin_t1)

  ((trainedModel, _), losses) <- mapAccumM [1..numEpochs] (initModel, optim) $ \epoc (model, opt) -> do
    let x_tensor = fmap (\t -> UnsafeMkTensor @ModelDevice @Dtype @'[1] $ D.asTensor [t]) sin_t
        t_tensor = fmap (\t -> UnsafeMkTensor @ModelDevice @Dtype @'[1] $ D.asTensor [t]) sin_t1
        prediction = sinRNN model x_tensor
        loss = meanSquaredError prediction t_tensor

    (newParam, newOp) <- runStep model opt loss 5e-2
    print $ "Training Loss: " ++ (show loss)
    return ((newParam, newOp), loss)

  let x_tensor = fmap (\t -> UnsafeMkTensor @ModelDevice @Dtype @'[1] $ D.asTensor [t]) sin_t
      pred = fmap (toDouble . (select @0 @0)) (sinRNN trainedModel x_tensor)
  plotWave "sinRNN_pred" (times, sin_t1) (times, pred)

  let x_tensor = fmap (\t -> UnsafeMkTensor @ModelDevice @Dtype @'[1] $ D.asTensor [t]) cos_t
      pred = fmap (toDouble . (select @0 @0)) (sinRNN trainedModel x_tensor)
  plotWave "sinRNN_eval" (times, cos_t1) (times, pred)
  return ()
