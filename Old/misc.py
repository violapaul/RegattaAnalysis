


################################################################
# Playing around with the J105 spinnaker rule

# SA = [(luff length + leech length) * .25 * foot length]
#    + [(half width – .5 * foot length) * (leech length + luff length)] ÷ 3
# where luff length shall not be greater than 15,100 mm nor less than 13,600 mm, leech
# length shall not be greater than 12,140 mm and half width shall not be less than .65 * foot length.

max_luff = 15.100
max_leech = 12.140

meter_per_foot = 3.048

print("max luff is", max_luff, max_luff * meter_per_foot)
print("max leech is", max_leech, max_leech * meter_per_foot)

def spinnaker_area (luff, leech, foot, half_width):
    one = ((luff + leech) * .25 * foot)
    two = ((half_width - 0.5 * foot) * (leech + luff))/ 3
    print(one, two, one+two)
