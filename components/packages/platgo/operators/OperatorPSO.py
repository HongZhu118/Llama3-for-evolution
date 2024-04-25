import numpy as np
from ..Population import Population


"""
 OperatorPSO - The operator of particle swarm optimization.
   Off = OperatorPSO(P,Pbest,Gbest) uses the operator of particle swarm
   optimization to generate offsprings based on particles P, personal best
   particles Pbest, and global best particles Gbest. P, Pbest, and Gbest
   should be arrays of Population objects, and Off is also an array of
   Population objects. Each object of P, Pbest, and Gbest is used to
   generate one offspring.

   Off = OperatorPSO(P,Pbest,Gbest,W) specifies the parameter of the
   operator, where W is the inertia weight.

   Example:
       Off = OperatorPSO(Population,Pbest,Gbest)

 ------------------------------------ Reference -------------------------------
 C. A. Coello Coello and M. S. Lechuga, MOPSO: A proposal for multiple
 objective particle swarm optimization, Proceedings of the IEEE Congress
 on Evolutionary Computation, 2002, 1051-1056.
 ------------------------------------------------------------------------------
"""


def OperatorPSO(Particle, Pbest, Gbest, *args):
    if len(args) > 0:
        W = args[0]
    else:
        W = 0.4
    ParticleDec = Particle.decs
    PbestDec = Pbest.decs
    GbestDec = Gbest.decs
    N, D = ParticleDec.shape
    ParticleVel = np.zeros((N, D))

    # Particle swarm optimization
    r1 = np.tile(np.random.random((N, 1)), (1, D))
    r2 = np.tile(np.random.random((N, 1)), (1, D))
    OffVel = W * ParticleVel + r1 * (PbestDec - ParticleDec) + r2 * (GbestDec - ParticleDec)  # noqa
    OffDec = ParticleDec + OffVel
    Offspring = Population(decs=OffDec, vel=OffVel)
    return Offspring
