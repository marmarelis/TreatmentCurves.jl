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

module TreatmentCurves

include("structures.jl")
include("predictor.jl")
include("constraints.jl")
include("weight-schemes.jl")
include("minimax.jl")
include("trustworthiness.jl")
include("potential-outcome.jl")
include("direct-training.jl")
include("neural-networks.jl")
include("analysis.jl")

export Instance, treat_instance, make_propensity_instance
export log_likelihood, ensemble_log_likelihood, train_directly!
export NeuralPredictor, ConstPredictor
export LomaxParam, BetaParam, GammaParam, ExpParam, BernoulliParam
export ConservationConstraint
export DirichletScheme, UnitScheme, BootstrapScheme, WeightCombo
export bound_expectation, maximize_expectation, lipschitz_condition
export generally_bound_ignorance_weights, bound_ignorance_weights
export BetaTrust, BernoulliTrust
export bound_potential_outcome
export ScaledKolmogorov, RationalSigmoid, LogKolmogorov, kolmogorov_mean
export bootstrap_prediction_statistic

end # module
