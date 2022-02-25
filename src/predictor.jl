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

# may or may not operate on a latent space from the covariates

using Flux
using Flux.Functors: @functor
# hitherto for differentiable models
abstract type Predictor{ In, Out, T<:Real } end

predict(::Predictor, ::AbstractVector) =
  error("unimplemented")
get_params(predictor::Predictor) = # ::Flux.Params
  Flux.params(predictor) # if @functor has been applied to the type


# idea: accommodate priors on the parametrization to do maximum a posteriori
#  ---with a hint of MaxEnt? favoring distributions of higher entropy

# predictor parametrizations
abstract type PredictionParam{ T<:Real } end

n_params(::PredictionParam) =
  error("unimplemented")
log_likelihood(::PredictionParam{T}, ::AbstractVector, ::T) where T =
  error("unimplemented")
sample_prediction(::PredictionParam{T}, ::AbstractVector) where T = # let the `AbstractVectors` containt `ForwardDiff.Dual`s
  error("optional unimplemented")
transform(::PredictionParam{T}, params::AbstractVector) where T =
  params

density(parametrization::PredictionParam{T}, params::AbstractVector, value::T) where T =
  log_likelihood(parametrization, params, value) |> exp
n_predictor_outputs(parametrization::PredictionParam, n_outputs::Int) =
  n_params(parametrization) * n_outputs
n_instance_outputs(::Predictor{In,POut}, parametrization::PredictionParam) where {In,POut} =
  div(POut, n_params(parametrization))

function log_likelihood(predictor::Predictor{In,POut,T},
    parametrization::PredictionParam{T}, instance::Instance{In,Out,T}
    )::T where {In,Out,POut,T}
  params = predict(predictor, instance.input) # do this in-place or as an SVector?
  param_size = n_params(parametrization)
  @assert Out * param_size == POut
  sum(instance.output|>enumerate) do (index, output)
    start = (index-1) * param_size + 1
    stop = start + param_size - 1
    slice = @view params[start:stop]
    trans_slice = transform(parametrization, slice)
    log_likelihood(parametrization, trans_slice, output)
  end
end

function map_prediction(func::Function, predictor::Predictor{In,POut,T},
    parametrization::PredictionParam{T}, input::SVector{In,T}) where {In,POut,T}
  params = predict(predictor, input)
  param_size = n_params(parametrization)
  Iterators.map(1:n_instance_outputs(predictor, parametrization)) do output_index
    start = (output_index-1) * param_size + 1
    stop = start + param_size - 1
    slice = @view params[start:stop]
    trans_slice = transform(parametrization, slice)
    func(trans_slice)
  end
end

function map_prediction(func::Function, predictor::Predictor{In,POut,T},
    parametrization::PredictionParam{T}, input::SVector{In}, index::Int) where {In,POut,T}
  params = predict(predictor, input)
  param_size = n_params(parametrization)
  start = (index-1) * param_size + 1
  stop = start + param_size - 1
  slice = @view params[start:stop]
  trans_slice = transform(parametrization, slice)
  func(trans_slice)
end


struct ConstPredictor{ In, Out, T } <: Predictor{In,Out,T}
  prediction :: MVector{Out, T}
end

ConstPredictor(In, Out, val) =
  ConstPredictor{In,Out,typeof(val)}(@MVector fill(val, Out))

@functor ConstPredictor

predict(model::ConstPredictor{In,Out,T}, ::AbstractVector) where {In,Out,T} =
  model.prediction


using Distributions
using Flux: softplus
using ChainRulesCore

# for SVector-type constructors that convert other `structure`s
function ChainRulesCore.rrule(constructor::Type{<:SArray}, structure)
  output = constructor(structure)
  project = ProjectTo(structure)
  pullback = tangent -> ( # `tangent` shares the structure (type?) of `output`
    NoTangent(), project(tangent) ) # basically convert tangent back to tuple or whatever
  output, pullback
end
#@Zygote.adjoint (V::Type{<:SArray})(x...) where T =
#  V(x...), y -> (y...,)

struct LomaxParam{ T } <: PredictionParam{T} end

n_params(::LomaxParam) = 2

function transform(::LomaxParam{T},
    (alpha, beta)::AbstractVector) where T
  softplus.((alpha, beta)) |> SVector{2} # represent logs directly? nah
end

function log_likelihood(::LomaxParam{T},
    (alpha, beta)::AbstractVector, value::T) where T
  log(alpha / beta) - (alpha+1) * log(1 + value/beta)
end

# exploit gamma-exponential mixture form
function sample_prediction(::LomaxParam{T}, (alpha, beta)::AbstractVector{T})::T where T
  gamma = Gamma(alpha, 1/beta)
  rate = rand(gamma)
  Exponential(1/rate) |> rand
end


# for a simple propensity model..
struct BetaParam{ T } <: PredictionParam{T}
  ceiling :: T
end

n_params(::BetaParam) = 2

using SpecialFunctions

function transform(param::BetaParam{T},
    (alpha, beta)::AbstractVector) where T
  curb = x -> param.ceiling * sigmoid(x/param.ceiling)
  curb.((alpha, beta)) |> SVector{2}
end

const beta_tolerance = 1e-2

function log_likelihood(::BetaParam{T},
    (alpha, beta)::AbstractVector, value::T) where T
  value = min(
    max(value,
      T(beta_tolerance) ),
    T(1-beta_tolerance) )
  #if value < beta_tolerance
  #  value += T(beta_tolerance)
  #elseif value > 1-beta_tolerance
  #  value -= T(beta_tolerance)
  #end
  (alpha-1)*log(value) + (beta-1)*log(1-value) - logbeta(alpha, beta)
end

function sample_prediction(::BetaParam{T}, (alpha, beta)::AbstractVector{T})::T where T
  Beta(alpha, beta) |> rand
end


struct GammaParam{ T } <: PredictionParam{T} end

n_params(::GammaParam) = 2

function transform(param::GammaParam{T},
    (alpha, beta)::AbstractVector) where T
  softplus.((alpha, beta)) |> SVector{2}
end

function log_likelihood(::GammaParam{T},
    (alpha, beta)::AbstractVector, value::T) where T
  alpha*log(beta) - loggamma(alpha) + (alpha-1)*log(value) - beta*value
end

function sample_prediction(::GammaParam{T}, (alpha, beta)::AbstractVector{T})::T where T
  Gamma(alpha, beta) |> rand
end


struct ExpParam{ T } <: PredictionParam{T} end

n_params(::ExpParam) = 1

function transform(::ExpParam{T},
    (lambda,)::AbstractVector) where T
  softplus.((lambda,)) |> SVector{1}
end

function log_likelihood(::ExpParam{T},
    (lambda,)::AbstractVector, value::T) where T
  log(lambda) - lambda*value
end

function sample_prediction(::ExpParam{T}, (lambda,)::AbstractVector{T})::T where T
  Exponential(1/lambda) |> rand
end


struct BernoulliParam{ T } <: PredictionParam{T}
  smoothness :: T
end

n_params(::BernoulliParam) = 1

function transform(param::BernoulliParam{T},
    (probability,)::AbstractVector) where T
  sigmoid.((probability/param.smoothness,)) |> SVector{1}
end

# make it more nuanced later on
discretize(::BernoulliParam, value) = value > 0.5

function log_likelihood(param::BernoulliParam{T},
    (probability,)::AbstractVector, value::T) where T
  event = discretize(param, value)
  if event
    log(probability)
  else
    log(1 - probability)
  end
end

function sample_prediction(::BernoulliParam{T},
    (probability,)::AbstractVector{T})::T where T
  Bernoulli(probability) |> rand |> T
end
