import logging

import numpy as np
import theano
import theano.tensor as tt
from pymc3.exceptions import ShapeError, DtypeError
from scipy.integrate import odeint

_log = logging.getLogger('pymc3')
floatX = theano.config.floatX


class DifferentialEquation(theano.Op):
    """
    Specify an ordinary differential equation

    .. math::
        \dfrac{dy}{dt} = f(y,t,p) \quad y(t_0) = y_0

    Parameters
    ----------

    func : callable
        Function specifying the differential equation. Must take arguments y (n_states,), t (scalar), p (n_theta,)
    times : array
        Array of times at which to evaluate the solution of the differential equation.
    n_states : int
        Dimension of the differential equation.  For scalar differential equations, n_states=1.
        For vector valued differential equations, n_states = number of differential equations in the system.
    n_theta : int
        Number of parameters in the differential equation.
    t0 : float
        Time corresponding to the initial condition

    .. code-block:: python

        def odefunc(y, t, p):
            #Logistic differential equation
            return p[0] * y[0] * (1 - y[0])

        times = np.arange(0.5, 5, 0.5)

        ode_model = DifferentialEquation(func=odefunc, times=times, n_states=1, n_theta=1, t0=0)
    """
    _itypes = [
        tt.TensorType(floatX, (False,)),  # y0 as 1D floatX vector
        tt.TensorType(floatX, (False,))  # theta as 1D floatX vector
    ]
    _otypes = [
        tt.TensorType(floatX, (False, False)),  # model states as floatX of shape (T, S)
        tt.TensorType(floatX, (False, False, False)),  # sensitivities as floatX of shape (T, S, len(y0) + len(theta))
    ]
    __props__ = ("func", "times", "n_states", "n_theta", "t0")

    def __init__(self, func, times, *, n_states, n_theta, t0=0):
        if not callable(func):
            raise ValueError("Argument func must be callable.")
        if n_states < 1:
            raise ValueError("Argument n_states must be at least 1.")
        if n_theta <= 0:
            raise ValueError("Argument n_theta must be positive.")

        # Public
        self.func = func
        self.t0 = t0
        self.times = tuple(times)
        self.n_times = len(times)
        self.n_states = n_states
        self.n_theta = n_theta
        self.n_p = n_states + n_theta

        # Private
        self._augmented_times = np.insert(times, 0, t0).astype(floatX)
        self._augmented_func = augment_system(func, self.n_states, self.n_theta)
        self._sens_ic = make_sens_ic(self.n_states, self.n_theta, floatX)

    def _system(self, Y, t, p):
        """This is the function that will be passed to odeint. Solves both ODE and sensitivities.

        Args:
            Y: augmented state vector (n_states + n_states + n_theta)
            t: current time
            p: parameter vector (y0, theta)
        """
        dydt, ddt_dydp = self._augmented_func(Y[:self.n_states], t, p, Y[self.n_states:])
        derivatives = np.concatenate([dydt, ddt_dydp])
        return derivatives

    def _simulate(self, y0, theta):
        # Initial condition comprised of state initial conditions and raveled sensitivity matrix
        s0 = np.concatenate([y0, self._sens_ic])

        # perform the integration
        sol = odeint(
            func=self._system, y0=s0, t=self._augmented_times, args=(np.concatenate([y0, theta]),),
        ).astype(floatX)
        # The solution
        y = sol[1:, :self.n_states]

        # The sensitivities, reshaped to be a sequence of matrices
        sens = sol[1:, self.n_states:].reshape(self.n_times, self.n_states, self.n_p)

        return y, sens

    def make_node(self, y0, theta):
        inputs = (y0, theta)
        states = self._otypes[0]()
        sens = self._otypes[1]()
        return theano.Apply(self, inputs, (states, sens))

    def __call__(self, y0, theta, return_sens=False, **kwargs):
        if isinstance(y0, (list, tuple)) and not len(y0) == self.n_states:
            raise ShapeError('Length of y0 is wrong.', actual=(len(y0),), expected=(self.n_states,))
        if isinstance(theta, (list, tuple)) and not len(theta) == self.n_theta:
            raise ShapeError('Length of theta is wrong.', actual=(len(theta),), expected=(self.n_theta,))

        # convert inputs to tensors (and check their types)
        y0 = tt.cast(tt.unbroadcast(tt.as_tensor_variable(y0), 0), floatX)
        theta = tt.cast(tt.unbroadcast(tt.as_tensor_variable(theta), 0), floatX)
        inputs = [y0, theta]
        for i, (input_val, itype) in enumerate(zip(inputs, self._itypes)):
            if not input_val.type == itype:
                raise ValueError(
                    'Input {} of type {} does not have the expected type of {}'.format(i, input_val.type, itype))

        # use default implementation to prepare symbolic outputs (via make_node)
        states, sens = super(theano.Op, self).__call__(y0, theta, **kwargs)

        # if theano.config.compute_test_value != 'off':
        #     # compute test values from input test values
        #     test_states, test_sens = self._simulate(
        #         y0=self._get_test_value(y0),
        #         theta=self._get_test_value(theta)
        #     )
        #
        #     # check types of simulation result
        #     if not test_states.dtype == self._otypes[0].dtype:
        #         raise DtypeError('Simulated states have the wrong type.', actual=test_states.dtype,
        #                          expected=self._otypes[0].dtype)
        #     if not test_sens.dtype == self._otypes[1].dtype:
        #         raise DtypeError('Simulated sensitivities have the wrong type.', actual=test_sens.dtype,
        #                          expected=self._otypes[1].dtype)
        #
        #     # check shapes of simulation result
        #     expected_states_shape = (self.n_times, self.n_states)
        #     expected_sens_shape = (self.n_times, self.n_states, self.n_p)
        #     if not test_states.shape == expected_states_shape:
        #         raise ShapeError('Simulated states have the wrong shape.', test_states.shape, expected_states_shape)
        #     if not test_sens.shape == expected_sens_shape:
        #         raise ShapeError('Simulated sensitivities have the wrong shape.', test_sens.shape, expected_sens_shape)
        #
        #     # attach results as test values to the outputs
        #     states.tag.test_value = test_states
        #     sens.tag.test_value = test_sens

        if return_sens:
            return states, sens
        return states

    def perform(self, node, inputs_storage, output_storage):
        y0, theta = inputs_storage[0], inputs_storage[1]
        # simulate states and sensitivities in one forward pass
        output_storage[0][0], output_storage[1][0] = self._simulate(y0, theta)

    # def infer_shape(self, node, input_shapes):
    #     s_y0, s_theta = input_shapes
    #     output_shapes = [(self.n_times, self.n_states), (self.n_times, self.n_states, self.n_p)]
    #     return output_shapes

    def grad(self, inputs, output_grads):
        _, sens = self.__call__(*inputs, return_sens=True)
        ograds = output_grads[0]

        # for each parameter, multiply sensitivities with the output gradient and sum the result
        # sens is (n_times, n_states, n_p)
        # ograds is (n_times, n_states)
        grads = [
            tt.sum(sens[:, :, p] * ograds)
            for p in range(self.n_p)
        ]

        # return separate gradient tensors for y0 and theta inputs
        result = tt.stack(grads[:self.n_states]), tt.stack(grads[self.n_states:])
        return result


def make_sens_ic(n_states, n_theta, floatX):
        """
        The sensitivity matrix will always have consistent form. (n_states, n_states + n_theta)

        If the first n_states entries of the parameters vector in the simulate call
        correspond to initial conditions of the system,
        then the first n_states columns of the sensitivity matrix should form
        an identity matrix.

        If the last n_theta entries of the parameters vector in the simulate call
        correspond to ode paramaters, then the last n_theta columns in
        the sensitivity matrix will be 0.

        Parameters
        ----------
        n_states : int
            Number of state variables in the ODE
        n_theta : int
            Number of ODE parameters
        floatX : str
            dtype to be used for the array

        Returns
        -------
        dydp : array
            1D-array of shape (n_states * (n_states + n_theta),), representing the initial condition of the sensitivities
        """

        # Initialize the sensitivity matrix to be 0 everywhere
        sens_matrix = np.zeros((n_states, n_states + n_theta), dtype=floatX)

        # Slip in the identity matrix in the appropirate place
        sens_matrix[:,:n_states] = np.eye(n_states, dtype=floatX)

        # We need the sensitivity matrix to be a vector (see augmented_function)
        # Ravel and return
        dydp = sens_matrix.ravel()
        return dydp


def augment_system(ode_func, n_states, n_theta):
    """
    Function to create augmented system.

    Take a function which specifies a set of differential equations and return
    a compiled function which allows for computation of gradients of the
    differential equation's solition with repsect to the parameters.

    Uses float64 even if floatX=float32, because the scipy integrator always uses float64.

    Parameters
    ----------
    ode_func : function
        Differential equation.  Returns array-like.
    n_states : int
        Number of rows of the sensitivity matrix. (n_states)
    n_theta : int
        Number of ODE parameters

    Returns
    -------
    system : function
        Augemted system of differential equations.
    """

    # Present state of the system
    t_y = tt.vector("y", dtype='float64')
    t_y.tag.test_value = np.ones((n_states,), dtype='float64')
    # Parameter(s).  Should be vector to allow for generaliztion to multiparameter
    # systems of ODEs.  Is m dimensional because it includes all initial conditions as well as ode parameters
    t_p = tt.vector("p", dtype='float64')
    t_p.tag.test_value = np.ones((n_states + n_theta,), dtype='float64')
    # Time.  Allow for non-automonous systems of ODEs to be analyzed
    t_t = tt.scalar("t", dtype='float64')
    t_t.tag.test_value = 2.459

    # Present state of the gradients:
    # Will always be 0 unless the parameter is the inital condition
    # Entry i,j is partial of y[i] wrt to p[j]
    dydp_vec = tt.vector("dydp", dtype='float64')
    dydp_vec.tag.test_value = make_sens_ic(n_states, n_theta, 'float64')

    dydp = dydp_vec.reshape((n_states, n_states + n_theta))

    # Get symbolic representation of the ODEs by passing tensors for y, t and theta
    yhat = ode_func(t_y, t_t, t_p[n_states:])
    # Stack the results of the ode_func into a single tensor variable
    if not isinstance(yhat, (list, tuple)):
        yhat = (yhat,)
    t_yhat = tt.stack(yhat, axis=0)

    # Now compute gradients
    J = tt.jacobian(t_yhat, t_y)

    Jdfdy = tt.dot(J, dydp)

    grad_f = tt.jacobian(t_yhat, t_p)

    # This is the time derivative of dydp
    ddt_dydp = (Jdfdy + grad_f).flatten()

    system = theano.function(
        inputs=[t_y, t_t, t_p, dydp_vec],
        outputs=[t_yhat, ddt_dydp],
        on_unused_input="ignore"
    )

    return system
