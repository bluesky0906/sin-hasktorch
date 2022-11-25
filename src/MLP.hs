{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleInstances #-}

module MLP where
import Torch.Typed 
import GHC.TypeNats
import GHC.Generics

data
  MLPSpec
    (inputFeatures :: Nat)
    (outputFeatures :: Nat)
    (hiddenFeatures :: Nat)
    (dtype :: DType)
    (device :: (DeviceType, Nat))
  = MLPSpec
  deriving (Eq, Show)

data
  MLP
    (inputFeatures :: Nat)
    (outputFeatures :: Nat)
    (hiddenFeatures :: Nat)
    (dtype :: DType)
    (device :: (DeviceType, Nat)) = MLP
  { layer0 :: Linear inputFeatures hiddenFeatures dtype device,
    layer1 :: Linear hiddenFeatures hiddenFeatures dtype device,
    layer2 :: Linear hiddenFeatures hiddenFeatures dtype device,
    layer3 :: Linear hiddenFeatures outputFeatures dtype device
  }
  deriving (Show, Generic, Parameterized)

instance
  (StandardFloatingPointDTypeValidation device dtype) =>
  HasForward
    (MLP inputFeatures outputFeatures hiddenFeatures dtype device)
    (Tensor device dtype '[batchSize, inputFeatures])
    (Tensor device dtype '[batchSize, outputFeatures])
  where
  forward MLP {..} = forward layer3 . Torch.Typed.tanh . forward layer2 . Torch.Typed.tanh . forward layer1 . Torch.Typed.tanh . forward layer0
  forwardStoch = (pure .) . forward

instance
  ( KnownDevice device,
    KnownDType dtype,
    All KnownNat '[inputFeatures, outputFeatures, hiddenFeatures],
    RandDTypeIsValid device dtype
  ) =>
  Randomizable
    (MLPSpec inputFeatures outputFeatures hiddenFeatures dtype device)
    (MLP inputFeatures outputFeatures hiddenFeatures dtype device)
  where
  sample MLPSpec =
    MLP <$> sample LinearSpec <*> sample LinearSpec <*> sample LinearSpec <*> sample LinearSpec
