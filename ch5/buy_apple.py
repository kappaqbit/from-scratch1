import layer_naive

if __name__ == "__main__":
    apple = 100
    apple_num = 2
    tax = 1.1

    mul_apple_layer = layer_naive.MulLayer()
    mul_tax_layer = layer_naive.MulLayer()

    apple_price = mul_apple_layer.forward(apple, apple_num)
    price = mul_tax_layer.forward(apple_price, tax)

    dprice = 1
    dapple_price, dtax = mul_tax_layer.backword(dprice)
    dapple, dapple_num = mul_apple_layer.backword(dapple_price)

    print("price:", int(price))
    print("dApple:", dapple)
    print("dApple_num:", int(dapple_num))
    print("dTax:", dtax)
