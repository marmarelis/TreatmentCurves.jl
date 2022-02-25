##  Copyright 2022 Myrl Marmarelis
##
##  Licensed under the Apache License, Version 2.0 (the "License");
##  you may not use this file except in compliance with the License.
##  You may obtain a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
##  Unless required by applicable law or agreed to in writing, software
##  distributed under the License is distributed on an "AS IS" BASIS,
##  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##  See the License for the specific language governing permissions and
##  limitations under the License.

# for bootstrap, Dirichlet reweighing, even test/train partitioning?
# deploy fast track for weights close to zero
abstract type WeightScheme{ T<:Real } end

draw_weights(::WeightScheme, n_draws::Int) =
  error("unimplemented")


struct WeightCombo{ T,
    A <: WeightScheme{T}, B <: WeightScheme{T} }
  first_scheme :: A
  second_scheme :: B
end
# run `reduce(WeightCombo, schemes)` ?

draw_weights(combo::WeightCombo, n_draws::Int) =
  ( draw_weights(combo.first_scheme, n_draws)
    .* draw_weights(combo.second_scheme, n_draws) )


struct UnitScheme{ T } <: WeightScheme{T} end

UnitScheme(T) = UnitScheme{T}()

draw_weights(scheme::UnitScheme{T}, n_draws::Int) where T =
  ones(T, n_draws)


struct DirichletScheme{ T,
    S <: WeightScheme{T} } <: WeightScheme{T}
  scale :: T
  pre_scheme :: S
end

DirichletScheme(scale::T) where T =
  DirichletScheme(scale, UnitScheme(T))

# https://news.ycombinator.com/item?id=30417811 see comments on bootstrapping
function draw_weights(scheme::DirichletScheme, n_draws::Int)
  pre_weights = draw_weights(scheme.pre_scheme, n_draws)
  ultimate_scale = sum(pre_weights)
  distribution = Dirichlet(scheme.scale .* pre_weights)
  observation_draws = rand(distribution, n_draws)
  draws = sum(observation_draws, dims=2)[:, 1]
  adjustment = ultimate_scale / n_draws
  draws * adjustment
end


struct BootstrapScheme{ T } <: WeightScheme{T} end

function draw_weights(scheme::BootstrapScheme{T}, n_draws::Int) where T
  weights = zeros(T, n_draws)
  weights[ sample(1:n_draws, n_draws) ] .+= 1
  weights
end
