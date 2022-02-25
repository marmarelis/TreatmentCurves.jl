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

using Flux, Zygote
using Random
using ProgressMeter
using StatsFuns

const Optimizer = Flux.Optimise.AbstractOptimiser

function ensemble_log_likelihood(predictors::Vector{P},
    parametrization::PredictionParam{T}, instance::Instance{In,Out,T},
    )::T where {In,Out,POut,T, P <: Predictor{In,POut,T}}
  ensemble_size = length(predictors)
  # `logsumexp` needs an intermediary array... (is reduce(logaddexp, ...) as stable?
  #   yes because logsumexp is implemented that way basically)
  log_likelihoods = Iterators.map(predictors) do predictor
    log_likelihood(predictor, parametrization, instance)
  end
  (reduce(logaddexp, log_likelihoods) - log(ensemble_size)) / Out
end

# INTERESTING THOUGHT, BUT UNFITTING
#  ensemble_size = length(predictors)
#  log_likelihoods = Iterators.map(zip(predictor_weights, predictors)) do (weights, predictor)
#    mean(zip(weights, instances)) do (weight, instance)
#      weight * log_likelihood(predictor, parametrization, instance)
#    end
#  end
#  (logsumexp(log_likelihoods) - log(ensemble_size)) / Out


using Base.Threads

# is "parameterization" more correct?
function train_directly!(predictors::Vector{P},
    instances::Vector{Instance{In,Out,T}}, parametrization::PredictionParam{T};
    weights::WeightScheme{T}, n_iterations::Int, optimizer::Optimizer, batch_size::Int,
    constraints::NTuple{C,Constraint{T}}=(), constraint_weight::T=T(0), n_constraint_draws::Int=0,
    validation_instances::Vector{Instance{In,Out,T}}=Instance{In,Out,T}[], verbose::Bool=false) where
      { In, Out, POut, C, T, P <: Predictor{In,POut,T} }
  @assert POut == n_predictor_outputs(parametrization, Out)
  @assert length(predictors) > 0
  sample_size = length(instances)
  validation_size = length(validation_instances)
  n_predictors = length(predictors)
  sample_indices = collect(1:sample_size) # filter for nonzero weights HERE
  predictor_weights = [
    draw_weights(weights, sample_size) for p in 1:n_predictors ]
  n_batches = div(sample_size, batch_size)
  n_constraints = length(constraints) # C
  CIs = [ constraint_instance_type(constraint, predictors[1], parametrization)
    for constraint in constraints ] # broadcasting doesn't seem to work here
  constraint_instances = [ Matrix{CI}(undef, n_constraint_draws, n_batches) for CI in CIs ]
  scores = zeros(T, n_iterations)
  validations = zeros(T, n_iterations)
  progress = Progress(n_iterations, enabled=verbose)
  for iteration in 1:n_iterations
    shuffle!(sample_indices)
    batch_indices = [
      @view sample_indices[(b-1)*batch_size+1 : b*batch_size]
      for b in 1:n_batches ]
    for (constraint_sample, constraint) in zip(constraint_instances, constraints)
      for draw in 1:n_constraint_draws, batch in 1:n_batches
        constraint_sample[draw, batch] = draw_instance(
          constraint, predictors[1], parametrization, instances)
      end
    end
    @threads for (weights, predictor) in # `collect` is needed for `@threads`
        zip(predictor_weights, predictors) |> collect
      Flux.train!(get_params(predictor),
          enumerate(batch_indices), optimizer) do batch, indices
        batch_weights = @view weights[indices]
        batch_instances = @view instances[indices]
        fit = mean(zip(batch_weights, batch_instances)) do (weight, instance)
          weight * log_likelihood(predictor, parametrization, instance)
        end
        penalty = n_constraints > 0 ? (
          mean(zip(constraint_instances, constraints)) do (constraint_sample, constraint)
            mean(@view constraint_sample[:, batch]) do constraint_instance
              apply_operator(
                constraint, predictor, parametrization, constraint_instance)
            end
          end) : T(0)
        -fit + constraint_weight*penalty
      end
    end
    score = mean(instances) do instance
      ensemble_log_likelihood(
        predictors, parametrization, instance)
    end
    validation = ( validation_size == 0 ? T(NaN) :
      mean(validation_instances) do instance
        ensemble_log_likelihood(
          predictors, parametrization, instance)
      end )
    scores[iteration] = score
    validations[iteration] = validation
    report = () -> [
      (:score, Float16(score)),
      (:validation, Float16(validation)) ]
    next!(progress, showvalues=report)
  end
  (; scores, validations, predictor_weights,
     predictors, instances, validation_instances ) # these ones are just returned back as passed in
end
