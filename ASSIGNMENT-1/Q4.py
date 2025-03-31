#4.1
for i in range(1,11):
    print(i)

print('\n')
#4.2
i=1
while i<=10:
    print(i)
    i+=1

print('\n')

#4.3
sum_for_loop=0
for i in range(1,101):
    sum_for_loop+=i
print(f'The sum of numbers from 1 to 100 using a for loop is: {sum_for_loop}')
sum_while_loop=0
i=1
while i<=100:
    sum_while_loop+=i
    i+=1
print(f'The sum of numbers from 1 to 100 using a while loop is: {sum_while_loop}')