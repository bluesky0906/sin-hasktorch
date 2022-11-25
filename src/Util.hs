
module Util where
import Graphics.Gnuplot.Simple
import Graphics.Gnuplot.Value.Tuple
import System.Process

sin_wave :: Double -> Double -> Double
sin_wave step_per_cycle x = Prelude.sin (x * (2 * (pi :: Double) / step_per_cycle))

cos_wave :: Double -> Double -> Double
cos_wave step_per_cycle x = Prelude.cos (x * (2 * (pi :: Double) / step_per_cycle))

plotWave ::
  C a =>
  String ->
  ([a], [a]) ->
  ([a], [a]) ->
  IO()
plotWave title (x1, y1) (x2, y2)  = do
  plotPaths [(EPS $ "img/" ++ title ++ ".eps"), (Title title)] [zip x1 y1, zip x2 y2]
  system $ "convert -background white -flatten -density 350 img/" ++ title ++ ".eps" ++ " -trim img/" ++ title ++ ".png"
  return ()