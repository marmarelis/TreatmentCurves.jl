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

abstract type TrustScheme{ T<:Real } end

"""
  Implement for particular parametrizations to the trust weights and propensity model.
"""
bound_ignorance_weights(::TrustScheme{T}, ::PredictionParam{T},
    propensity_params::AbstractVector{T}, order::Int,
    sensitivity::T, treatment::T) where T = # returns (lower, upper)
  error("unimplemented")

"""
  Evaluate the weighing function centered at `treatment` on `point`.
  Possibly (probably) unnormalized for each `treatment`.
"""
weigh_treatment(::TrustScheme{T}, treatment::T, point::T) where T =
  error("unimplemented")

# may not actually be needed, depending on how the general method is implemented
integrate_weights(::TrustScheme{T}, treatment::T) where T =
  error("unimplemented")


"""
  Monte Carlo approximator of confounded ignorance bounds for any `TrustScheme` and propensity.
"""
function generally_bound_ignorance_weights(scheme::TrustScheme{T}, param::PredictionParam{T},
    propensity_params::AbstractVector{T}, order::Int, sensitivity::T, treatment::T;
    n_draws::Int)::NTuple{2,T} where T
  @assert sensitivity > 1
  @assert order in (1, 2)

end


using HypergeometricFunctions, SpecialFunctions

struct BetaTrust{ T } <: TrustScheme{T}
  # (alpha + beta := r) - 2, must be greater than zero
  precision :: T
end

# idea: decouple expectations (except for the first one out of three)
#   from the gamma-bound parts?

function bound_ignorance_weights(scheme::BetaTrust{T}, ::BetaParam{T},
    (prop_alpha, prop_beta)::AbstractVector{T}, order::Int,
    sensitivity::T, treatment::T)::NTuple{2,T} where T
  @assert sensitivity > 1
  @assert order in (1, 2)
  log_gamma = log(sensitivity)
  (hyper_up, hyper_lo) = _₁F₁.(
    scheme.precision * treatment + prop_alpha,
    scheme.precision + prop_alpha + prop_beta,
    (+log_gamma, -log_gamma) ) # \plusminus logGamma
  first_scale = sensitivity^treatment * log_gamma
  first_expectation = (
    ((1-treatment) * prop_alpha - treatment * prop_beta)
    / (prop_alpha + prop_beta + scheme.precision) )
  (first_up, first_lo) = (+1, -1) .* abs(first_scale * first_expectation)
  if order == 1
    (hyper_lo - first_up, hyper_up - first_lo)
  else
    # THIS expectation cannot be negative
    second_expectation = ((
        (prop_alpha^2 + prop_beta^2 + 2prop_alpha*prop_beta
          + prop_alpha + prop_beta - scheme.precision) * treatment^2
        - (2prop_alpha^2 + 2prop_alpha*prop_beta
          + 2prop_alpha - scheme.precision) * treatment
        + prop_alpha * (prop_alpha+1)
      ) / (
        (prop_alpha + prop_beta + scheme.precision)
        * (prop_alpha + prop_beta + scheme.precision + 1) ))
    @assert second_expectation >= 0
    second_scale_up = log_gamma^2 * sensitivity^treatment
    second_scale_lo = T(0)
    (second_up, second_lo) =
      (second_scale_up, second_scale_lo) .* second_expectation
    (both_up, both_lo) = (first_up+second_up, first_lo+second_lo)
    (hyper_lo - both_up, hyper_up - both_lo)
  end
end
# possibly make BetaParam an abstract type grounded by different transformations?

function weigh_treatment(scheme::BetaTrust{T}, treatment::T, point::T)::T where T
  first_part = point ^ (scheme.precision * treatment) # a-1 = (r-2)t
  second_part = (1-point) ^ (scheme.precision * (1-treatment))
  first_part * second_part
end


# do NOT use the general approximator on this. baseline below...
struct BernoulliTrust{ T } <: TrustScheme{T}
  # ways to control *how* stochastic or not?
  # thresholding versus sampling versus "Laplace rule" type debiasing...
end

# what about naively plugging in the continuous densities for the bernoulli propensity value?
function bound_ignorance_weights(scheme::BernoulliTrust{T},
    parametrization::BernoulliParam{T}, (probability,)::AbstractVector{T},
    order::Int, sensitivity::T, treatment::T)::NTuple{2,T} where T
  event = discretize(parametrization, treatment)
  propensity = event ? probability : (1-probability)
  # due to Jesson et al. (2021)
  upper = sensitivity / (propensity*(sensitivity-1) + 1)
  lower = 1 / (propensity*(1-sensitivity) + sensitivity)
  (lower, upper)
end

function bound_ignorance_weights(scheme::BernoulliTrust{T},
    parametrization::BetaParam{T}, (alpha, beta)::AbstractVector{T},
    order::Int, sensitivity::T, treatment::T)::NTuple{2,T} where T
  probability = alpha / (alpha + beta)
  bound_ignorance_weights(scheme, BernoulliParam(T(0)),
    (probability,) |> SVector{1}, order, sensitivity, treatment)
end
