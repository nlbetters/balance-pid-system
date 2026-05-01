import math
import time

MIN_DT = 0.001
MAX_DT = 0.05
INTEGRAL_LIMIT = 250.0
DERIVATIVE_LIMIT = 5000.0

class PIDcontroller:
    def __init__(self, kp, ki, kd, alpha, beta, max_theta, conversion="linear"): #"linear" or "tanh"

        self.kp, self.ki, self.kd = kp, ki, kd
        self.alpha = alpha  #Exponential Filter: α⋅x + (1-α)⋅x_last
        self.beta = beta  #Coefficient for converting magnitude, either βx or tanh(βx)
        self.max_theta = max_theta

        if conversion == "linear":
            self.magnitude_convert = 1 #Linear
        elif conversion == "tanh":
            self.magnitude_convert = 0 #Tanh
        else:
            self.magnitude_convert = -1
 
        self.prev_out_x = 0.0
        self.prev_err_x = 0.0  
        self.prev_out_y = 0.0
        self.prev_err_y = 0.0

        self.sum_err_x = 0.0  #Integral
        self.sum_err_y = 0.0  #Integral
        
        self.last_time = None
        self.has_previous_error = False

    def pid(self, target, current):

        # Time step. Clamp dt so derivative does not spike if the loop timing jitters.
        new_time = time.perf_counter()
        if self.last_time is None:
            dt = MIN_DT
        else:
            dt = new_time - self.last_time
            dt = max(MIN_DT, min(dt, MAX_DT))

        #errors
        err_x = current[0] - target[0]
        err_y = current[1] - target[1]
        self.sum_err_x += err_x * dt
        self.sum_err_y += err_y * dt
        self.sum_err_x = max(-INTEGRAL_LIMIT, min(self.sum_err_x, INTEGRAL_LIMIT))
        self.sum_err_y = max(-INTEGRAL_LIMIT, min(self.sum_err_y, INTEGRAL_LIMIT))

        if self.has_previous_error:
            d_err_x = (err_x - self.prev_err_x) / dt
            d_err_y = (err_y - self.prev_err_y) / dt
            d_err_x = max(-DERIVATIVE_LIMIT, min(d_err_x, DERIVATIVE_LIMIT))
            d_err_y = max(-DERIVATIVE_LIMIT, min(d_err_y, DERIVATIVE_LIMIT))
        else:
            d_err_x = 0.0
            d_err_y = 0.0

        #output
        pid_x = self.kp * err_x + self.ki * self.sum_err_x + self.kd * d_err_x
        pid_y = self.kp * err_y + self.ki * self.sum_err_y + self.kd * d_err_y
        filtered_x = self.alpha * pid_x + (1 - self.alpha) * self.prev_out_x
        filtered_y = self.alpha * pid_y + (1 - self.alpha) * self.prev_out_y
        
        #Convert to spherical coordinates
        phi = math.degrees(math.atan2(filtered_y, filtered_x))
        if phi < 0:
            phi += 360
        r = math.sqrt(filtered_x**2 + filtered_y**2)
        if self.magnitude_convert == 1:
            theta = min(max(0, self.beta*r), self.max_theta)
        else:
            theta = max(0, self.max_theta * math.tanh(self.beta*r))


        self.prev_err_x = err_x
        self.prev_err_y = err_y
        self.prev_out_x = filtered_x
        self.prev_out_y = filtered_y
        self.last_time = new_time
        self.has_previous_error = True

        return theta, phi #in degrees

    def reset(self):
        self.prev_out_x = 0.0
        self.prev_err_x = 0.0
        self.prev_out_y = 0.0
        self.prev_err_y = 0.0
        self.sum_err_x = 0.0
        self.sum_err_y = 0.0
        self.last_time = None
        self.has_previous_error = False
