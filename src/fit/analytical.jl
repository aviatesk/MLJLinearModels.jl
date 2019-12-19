# Solvers corresponding to solving a closed-form expression

"""
$SIGNATURES

Fit a least square regression either with no penalty (OLS) or with a L2 penalty (Ridge).

## Complexity

Assuming `n` dominates `p` so that `np²` dominates `p³`,

* non-iterative (full solve):     O(np²) - dominated by the construction of the Hessian X'X.
* iterative (conjugate gradient): O(κnp) - with κ the number of CG steps (κ ≤ p).
"""
function _fit(glr::GLR{L2Loss,<:L2R}, solver::Analytical, X, y)
	# full solve
	if !solver.iterative
		λ  = getscale(glr.penalty)
		if iszero(λ)
			# standard LS solution
			return augment_X(X, glr.fit_intercept) \ y
		else
			# Ridge case -- form the Hat Matrix then solve
			H = form_XtX(X, glr.fit_intercept, λ)
			b = X'y
			glr.fit_intercept && (b = vcat(b, sum(y)))
			return cholesky!(H) \ b
		end
	end
	# Iterative case, note that there is no augmentation here
	# it is done implicitly in the application of the Hessian to
	# avoid copying X
	# The number of CG steps to convergence is at most `p`
	p = size(X, 2) + Int(glr.fit_intercept)
	max_cg_steps = min(solver.max_inner, p)
	# Form the Hessian map, cost of application H*v is O(np)
	Hm = LinearMap(Hv!(glr, X, y), p; ismutating=true, isposdef=true, issymmetric=true)
	b  = X'y
	glr.fit_intercept && (b = vcat(b, sum(y)))
	return cg(Hm, b; maxiter=max_cg_steps)
end

# TODO: pass an evaluator (criterion)
# TODO: pass argument to check whether need to return all θ(λ) or not
# TODO: analytical should be able to take specs like QR/LU/SVD
# TODO: copy randomized SVD from LowRankModels and allow passing here
#
# function _fitcv(glrcv::GLRCV{GLR{L2Loss, L2R}}, solver::Analytical, X, y)
# 	n, p = size(X)
# 	if n < p
# 		# fat case
# 		return _ridgecv_svd_fat(glrcv, X, y)
#     end
# 	# n >= p, tall case
# 	return _ridgecv_svd_tall(glrcv, X, y)
# end
#
# _ridgecv_gcv()
#
# function _ridgecv_svd_fat(glrcv, X, y)
# 	# Initial computation
# 	M = Symmetric(Xfat * Xfat') # O(pn²) to construct
# 	F = svd(M) 				    # O(κn³) to compute
# 	U = F.U
# 	S = F.S
#     # Now compute θ(λ) = X'UD(λ)U'y where D(λ) = (Σ̃² + λI)⁻¹
#
# end
#
# function _ridgecv_svd_tall(glrcv, X, y)
# end
