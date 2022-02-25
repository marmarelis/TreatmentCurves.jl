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

abstract type KolmogorovFunction end

const sigmoid_limit = 1e8 # on the pre-scaled value

struct RationalSigmoid <: KolmogorovFunction end
# assumes non-negative. more generally, take abs and some power
forward(::RationalSigmoid, x) = x > sigmoid_limit ? 1 : x/(1+x) # appropriate casting happens automagically here
backward(::RationalSigmoid, y) = y/(1-y)


struct LogKolmogorov <: KolmogorovFunction end

forward(::LogKolmogorov, x) = log1p(x) # actually Log1pKolmogorov...
backward(::LogKolmogorov, y) = exp(y) - 1


struct ScaledKolmogorov{
    F <: KolmogorovFunction, T <: Real } <: KolmogorovFunction
  scale :: T
  inner :: F
end

forward(f::ScaledKolmogorov{F,T}, x::T) where {F,T} = forward(f.inner, x/f.scale)
backward(f::ScaledKolmogorov{F,T}, y::T) where {F,T} = backward(f.inner, y) * f.scale


function kolmogorov_mean(f::KolmogorovFunction, iterable, weights) # `iterator` means something else; it is the instance we use to iterate
  backward(f,
    sum(((x, w),) -> forward(f, x) * w, zip(iterable, weights))
      / sum(weights) )
end


# should I allow medians as well??
function bootstrap_prediction_statistic(statistic::Function, predictors::Vector{P},
    instances::AbstractVector{Instance{In,Out,T}}, parametrization::PredictionParam{T};
    kolmogorov::KolmogorovFunction, weight_scheme::WeightScheme{T}, n_resamples::Int # `mean_function` means something specific too, like a mu(x) model
    )::Matrix{T} where {In,Out,POut,T, P <: Predictor{In,POut,T}}
  ensemble_size = length(predictors)
  n_instances = length(instances) # do I resample over all (predictors x instances)
  evaluations = reduce(hcat, Iterators.map( # n_predictions (ie n_statistics) x n_evaluations
      Iterators.product(predictors, instances)) do (predictor, instance)
    map_prediction(statistic, predictor, parametrization, instance.input) |> collect
  end )
  n_statistics, n_evaluations = size(evaluations)
  reduce(hcat, Iterators.map(1:n_resamples) do trial
    #weights = draw_weights(weight_scheme, n_evaluations)
    predictor_weights = draw_weights(weight_scheme, ensemble_size)
    instance_weights = draw_weights(weight_scheme, n_instances) # allow decoupling of schemes
    weights = reshape(predictor_weights * instance_weights', :)
    #reweighted = ifelse.(weights' .== 0, T(0), # must reweigh within the sum, actually
    #  weights' .* evaluations ) # ensure Inf * 0 is zero, not NaN
    result = mapslices(evaluations, dims=2) do sample
      kolmogorov_mean(kolmogorov, sample, weights)
    end
    @view result[:, 1]
  end )
end
