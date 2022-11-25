{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}


module SinMLP where
import MLP
import Util
import Torch.Typed
import qualified Torch.Tensor as D
import Torch.Control (mapAccumM)
import Graphics.Gnuplot.Simple
import GHC.Generics
import GHC.TypeNats
import GHC.Exts
import System.Process


step_per_cycle :: Double 
step_per_cycle = 1000

number_of_cycles :: Double
number_of_cycles = 1

type Dtype = 'Double
type ModelDevice = '( 'CPU, 0)
type InputSize = 200
type HiddenSize = 100


main :: IO()
main = do
  let times = [0..(step_per_cycle * number_of_cycles - 1)]
      sin_t = fmap (sin_wave step_per_cycle) times
      sin_t1 = tail sin_t ++ [sin_wave step_per_cycle (step_per_cycle * number_of_cycles)]
      -- テストデータ
      cos_t = fmap (cos_wave step_per_cycle) times
      cos_t1 = tail cos_t ++ [cos_wave step_per_cycle (step_per_cycle * number_of_cycles)]
  initModel <- sample (MLPSpec @1 @1 @HiddenSize @Dtype @ModelDevice)
  -- | setting
  let numEpochs = 200
      optim = mkAdam 0.001 0.9 0.999 (flattenParameters initModel)
      -- optim = GD
  plotPaths [(EPS "img/sin.eps"), (Title "sin t and sint+1")] [zip times sin_t, zip times sin_t1]
  system "convert -background white -flatten -density 350 img/sin.eps -trim img/sin.png"

  ((trainedModel, _), losses) <- mapAccumM [1..numEpochs] (initModel, optim) $ \epoc (model, opt) -> do
    let x_tensor = UnsafeMkTensor @ModelDevice @Dtype @'[InputSize, 1] $ D.asTensor $ fmap (\t -> [t]) sin_t
        t_tensor = UnsafeMkTensor @ModelDevice @Dtype @'[InputSize, 1] $ D.asTensor $ fmap (\t -> [t]) sin_t1
        y_tensor = forward model x_tensor
    let loss = mseLoss @ReduceMean y_tensor t_tensor
    u <- runStep model opt loss 10e-3
    print $ "Training Loss: " ++ (show loss)
    return (u, loss)
  -- テスト
  let x_tensor = UnsafeMkTensor @ModelDevice @Dtype @'[InputSize, 1] $ D.asTensor $ fmap (\t -> [t]) sin_t
      y = fmap (\x -> x !! 0) $ toList (Just (forward trainedModel x_tensor))
  plotPaths [(EPS "img/sinMLP_pred.eps"), (Title "sin t+1 and pred")] [zip times sin_t1, zip times y]
  system "convert -background white -flatten -density 350 img/sinMLP_pred.eps -trim img/sinMLP_pred.png"

  let x_tensor = UnsafeMkTensor @ModelDevice @Dtype @'[InputSize, 1] $ D.asTensor $ fmap (\t -> [t]) cos_t
      y = fmap (\x -> x !! 0) $ toList (Just (forward trainedModel x_tensor))
  plotPaths [(EPS "img/sinMLP_eval.eps"), (Title "cos t+1 and pred")] [zip times cos_t1, zip times y]
  system "convert -background white -flatten -density 350 img/sinMLP_eval.eps -trim img/sinMLP_eval.png"
  return ()

