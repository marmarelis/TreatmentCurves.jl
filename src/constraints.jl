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

using ForwardDiff

abstract type Constraint{ T <: Real } end # do I really need the `T` for the supertype?

draw_instance(::Constraint{T}, ::Predictor{In,POut,T}, ::PredictionParam{T},
    ::Vector{Instance{In,Out,T}}) where {In,POut,Out,T} = #::ConstraintInstance{PIn,Out,T}
  error("unimplemented")

apply_operator(::Constraint{T}, ::Predictor{In,POut,T}, ::PredictionParam{T},
    ::ConstraintInstance{In,COut,T}) where {In,POut,COut,T} =
  error("unimplemented")

constraint_instance_type(::Constraint{T}, ::PredictionParam{T}, ::Predictor{In,POut,T}
    ) where {In,POut,T} =
  error("unimplemented")


struct ConservationConstraint{ T } <: Constraint{T}
  input_indices :: Matrix{Int} # batched as columns so that gradient only has to be taken once
  coefficients :: Matrix{T}
  positive :: Vector{Bool} # per column
  absolute :: Vector{Bool}
  mimicked_indices :: Vector{Int} # indices to grab from a random instance; recycled
end

constraint_target(::LomaxParam, (alpha, beta)::AbstractVector) =
  (alpha - 1) / beta # mode of the rate parameter

@generated function constraint_instance_type(::ConservationConstraint{T},
    ::Predictor{In,POut,T}, ::PredictionParam{T}) where {In,POut,T} # make this a macro? (harder to know types of arguments)
  ConstraintInstance{In,0,T} # do I need to quote this even though it's just a type value?
end

function draw_instance(constraint::ConservationConstraint{T}, predictor::Predictor{In,POut,T},
    parametrization::PredictionParam{T}, instances::Vector{Instance{In,Out,T}}
    )::ConstraintInstance{In,0,T} where {In,POut,Out,T}
  grabbed_instance = sample(instances)
  input = @MVector rand(T, In)
  for i in constraint.mimicked_indices # just overwrite previously randomized
    input[i] = grabbed_instance.input[i]
  end
  # I guess I could always subclass the ConstraintInstance?
  # generic structure seems to work fine for now.
  ConstraintInstance(
    convert(SVector, input),
    SVector{0,T}() )
end

function apply_operator(constraint::ConservationConstraint{T}, predictor::Predictor{In,POut,T},
    parametrization::PredictionParam{T}, instance::ConstraintInstance{In,0,T})::T where {In,POut,T}
  n_indices, n_laws = size(constraint.input_indices)
  @assert (n_indices, n_laws) == size(constraint.coefficients)
  n_targets = n_instance_outputs(predictor, parametrization)
  sum(1:n_targets) do target_index
    # recent versions of Zygote can ALMOST differentiate their own gradients
    derivatives = ForwardDiff.gradient(instance.input) do input # input -> begin ... end
      map_prediction(params -> constraint_target(parametrization, params),
        predictor, parametrization, input, target_index)
    end
    #derivatives, = grad(central_fdm(diff_order, 1), differentiable, instance.input)
    sum(1:n_laws) do law
      operator = sum(1:n_indices) do index
        input = constraint.input_indices[index, law]
        coefficient = constraint.coefficients[index, law]
        metric = derivatives[input]
        if constraint.absolute[law]
          metric = abs(metric)
        end
        coefficient * metric
      end
      if constraint.positive[law]
        min(operator, 0) ^ 2
      else
        operator ^ 2 # penalize nonzero-ness
      end
    end
  end
end
