def verified_luhn(number):
    number = number.replace(" ", "")
    total = 0
    inverse_digits = number[::-1]

    for i, chiffre in enumerate(inverse_digits):
        n = int(chiffre)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n

    return total % 10 == 0


card_number = "4012 8888 8888 1882"
if verified_luhn(card_number):
    print("✅ valid card number")
else:
    print("❌ invalid card number")



