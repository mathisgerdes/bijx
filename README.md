# Crouch Grossmann integration in JAX

General implementation of CG solvers:
- Can specify general Butcher tableau.
- Integrand can be any pytree containing either real or GL(N) valued arrays.
- JAX autodiff supported via the adjoint sensitivity method.

Primary usage through the function `crouch_grossmann(vector_field, y0, args, t0, t1, step_size, is_lie, tableau=EULER)`.
The argument `is_lie` must be a pytree matching the structure of `y0` but with leaves (values) booleans (True/False) indicating whether the corresponding array in `y0` is in $GL(N)$ or real.

The ODE must be defined in terms of a function `vector_field(t, y, args)`.
Outputs must match the structure of `y0`.
In the case of $GL(N)$, tangent vectors must be specified as lie algebra elements, i.e. as vectors at the identity element.
Currently, transport is then defined via right multiplication (i.e. $AU$ if $A$ is in the lie algebra and $U$ is the group element).
