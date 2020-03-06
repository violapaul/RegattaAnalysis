"""
Computing the thermal effects on mast tension.

Note, each turn of a large turnbuckle is 0.050 inches.

"""


length = 40 * 12 # inches

aluminum_coefficient = 13.0 / 10**6  # expansion per degree F
steel_coefficient = 7.2 / 10**6  # expansion per degree F

delta_temperature = 15

delta_aluminum = delta_temperature * length * aluminum_coefficient
delta_steel    = delta_temperature * length * steel_coefficient

print(delta_steel - delta_aluminum)
