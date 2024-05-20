# Gradient Descent
import numpy as np
import matplotlib.pyplot as plt

# data_points_x = []
# data_points_y = []

def gradient_descent(x, y, step_size = 0.001, tolerance = 0.00001):
	m = 0
	b = 0
	step = np.array([1,1])
	times = 0
	while np.linalg.norm(step) > tolerance:
		step = np.array(get_step(m,b,x,y))
		m -= step[0] * step_size
		b -= step[1] *step_size
		times += 1
		# data_points_x.append(times)
		# data_points_y.append(np.linalg.norm(m*x+b-y))
	return m, b

def get_step(m,b,x,y):
	v = np.array([1 for _ in range (len(x))])
	return (2*m*np.dot(x,x) + 2*b*np.dot(x,v) - 2*np.dot(x,y)), (2*m*np.dot(x,v) - 2*np.dot(y,v) + 2 * b * np.dot(v,v))

def netwon_method(x, y):
	point = np.array([0.0,0.0])
	v = np.array([1 for _ in range(len(x))])

	hessian = np.array([[2 * np.dot(x,x), 2 * np.dot(x,v)],[2 * np.dot(x,v), 2 * np.dot(v,v)]])
	gradient = np.array(get_step(*point,x,y))
	step = np.linalg.inv(hessian) @ gradient

	while(np.linalg.norm(step) > 0.0000001):
		point -= step
		gradient = np.array(get_step(*point, x, y))
		hessian = np.array([[2 * np.dot(x,x), 2 * np.dot(x,v)],[2 * np.dot(x,v), 2 * np.dot(v,v)]])
		step = np.linalg.inv(hessian)*gradient
	return point

x = np.array(list(map(float, input().split())))
y = np.array(list(map(float, input().split())))

m, b = gradient_descent(x, y)
print(m, b)

m, b = netwon_method(x, y)
print(m, b)


# # Create a plot
# fig, ax = plt.subplots()

# ax.plot(data_points_x, data_points_y)

# ax.set_xlabel('steps')
# ax.set_ylabel('error')

# # Show the plot
# plt.show()