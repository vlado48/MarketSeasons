import numpy as np

def bisection(f, a, b, N, *args):
    fa, fb = f(a, *args), f(b, *args)
    if fa*fb<0:        
        for n in range(N):
            c = 0.5*(a+b)
            fc = f(c, *args)
            if fc*fa<0:
                b = c
            else:
                a = c
        return c, fc
    else:
        print(f'there is no solution for {f.__name__} in interval {{a, b}}')
        return 0, 0
    
def newton_method(F, x, N=100, args=(), xtol=1e-8):
    for n in range(N):
        f, df = F(x, *args)
        dx = f/df
        if np.abs(dx)<xtol: 
            print(f'Convergence limit has been achieved in {n} iterations.')
            break
        elif n==N-1:
            print(f'After {N} iterations convergence of {np.abs(dx)} has been achieved.')
        x -= dx
    return (x, f, dx, n)  