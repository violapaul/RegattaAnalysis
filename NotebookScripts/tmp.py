

shares = [
    (510.4910, 195.89),
    (522.4930, 191.39),
    (525.9280, 190.14),
    (543.9510, 183.84),
    (283.2860, 176.50),
    (382.3800, 130.76)
]

price = 228.75
total = 0
price_basis = 0

for n, p in shares:
    print(n, p)
    total += n
    price_basis += n * p

    
print(total, total*price, price_basis)
