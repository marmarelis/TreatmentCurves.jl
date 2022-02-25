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

using Base.Threads
using ProgressMeter

# integrate with IPW to attain unbiased overall dose-response curves?
function bound_potential_outcome(operate::Function, # monotonic
    covariates::AbstractVector{SVector{CIn,T}},
    predictors::Vector{P}, predictor_param::PredictionParam{T},
    propensity_models::Vector{PP}, propensity_param::PredictionParam{T},
    weight_scheme::WeightScheme{T}, trust_scheme::TrustScheme{T},
    treatment_index::Int; n_resamples::Int, n_draws::Int, order::Int, verbose::Bool=true,
    sensitivity::T, treatment::T, generator::Distribution, static_draws::AbstractVector{T}=T[],
    weight_threshold::T=T(1e-8) )::NTuple{2,Array{T,3}} where {
      In, CIn, POut, PPIn, PPOut, T, P <: Predictor{In,POut,T}, PP <: Predictor{PPIn,PPOut,T} } # In-1 is invalid syntactically
  @assert In == CIn+1
  n_predictors = length(predictors)
  n_propensities = length(propensity_models)
  n_instances = length(covariates)
  treatment_index = wrap_treatment_index(treatment_index, In)
  instances = [ Instance{In, 0, T}(i,
    vcat(
        covariate[1:treatment_index-1],
        [treatment],
        covariate[treatment_index:end]
      ) |> SVector{In,T},
    SVector{0,T}([]) ) for (i, covariate) in enumerate(covariates) ]
  n_outputs = n_instance_outputs(predictors[1], predictor_param)
  progress = Progress(n_propensities * n_instances, desc="Evaluating.. ", enabled=verbose)
  denominator_bounds = map(Iterators.product(propensity_models, instances)) do (model, instance)
    next!(progress)
    propensity_instance = make_propensity_instance(instance, treatment_index)
    parameters = map_prediction(
      identity, model, propensity_param, propensity_instance.input, 1)
    bounds = bound_ignorance_weights(trust_scheme, propensity_param,
      parameters, order, sensitivity, treatment)
  end
  denominators_up = [b[2] for b in denominator_bounds]
  denominators_lo = [b[1] for b in denominator_bounds]
  # replicate `bootstrap_prediction_statistic` here?
  bootstrapped_upper = fill(T(NaN), n_outputs, n_resamples, n_instances)
  bootstrapped_lower = fill(T(NaN), n_outputs, n_resamples, n_instances)
  progress = Progress(n_resamples*n_instances, desc="Bootstrapping.. ", enabled=verbose) # better hope this is thread-safe
  @threads for i in 1:n_resamples
    # enable e.g. Bernoulli draws to simply cover the entire support {0,1} once
    draws = length(static_draws) > 0 ? static_draws : convert.(T, rand(generator, n_draws))
    functions = operate.(draws)
    generator_densities = convert.(T, pdf.(generator, draws)) # a la importance sampling
    # this and `denominator_bounds` are (models x instances) matrices
    numerators = map(Iterators.product(predictors, instances)) do (model, instance)
      next!(progress)
      densities = map_prediction(model, predictor_param, instance.input) do params
        # apply the same sample to every output dimension
        [ density(predictor_param, params, draw) / scale
          for (draw, scale) in zip(draws, generator_densities) ]
      end
      reduce(hcat, densities) # draws x outputs
    end # matrix of matrices
    # the block above, up to the loop beginning, were heretofore not in the resampler.
    # but this hopefully averages out some of the monte carlo biases
    predictor_weights = draw_weights(weight_scheme, n_predictors)
    propensity_weights = draw_weights(weight_scheme, n_propensities)
    # I realized that it's permitted to do vector math on a matrix of matrices! (as I do in the following line)
    numerator = (permutedims(numerators) * predictor_weights) ./ sum(predictor_weights)
    denominator_up = (denominators_up' * propensity_weights) ./ sum(propensity_weights)
    denominator_lo = (denominators_lo' * propensity_weights) ./ sum(propensity_weights)
    for k in 1:n_outputs, j in 1:n_instances
      ratios = @view numerator[j][:, k]
      expectee = functions
      if denominator_lo[j] > weight_threshold
        upper = 1/denominator_lo[j]
        lower = 1/denominator_up[j]
        result = bound_expectation(
          ratio -> ratio .* (lower, upper), ratios, expectee)
        bootstrapped_upper[k, i, j] = result[2]
        bootstrapped_lower[k, i, j] = result[1]
      end
      if k == n_outputs
        report = () -> []
        next!(progress, showvalues=report)
      end
    end
  end
  (bootstrapped_lower, bootstrapped_upper)
end

# for comparison to the binary-treatment solutions of Kallus or Jesson,
# repeat the SAME procedure just with binary propensities on roughly discretized treatments.
# the above need not be modified at all?
