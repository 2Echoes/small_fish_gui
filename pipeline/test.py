def yielder(n) :
    for i in range(n) : 
        yield i

def constructor(n) :
    def inner() :
        return yielder(n)
    return inner

#test

N = 10

gen1 = yielder(N)

a = constructor(10)

print(type(a))

gen2 = a()
gen3 = a()

print('gen1 : ', list(gen1))
print('gen2 : ', list(gen2))
print('gen3 : ', list(gen3))
print('gen3 : ', list(gen3))
