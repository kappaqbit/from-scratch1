import layer_naive

if __name__ == "__main__":
    apple = 100
    apple_num = 2
    orange = 150
    orange_num = 3
    tax = 1.1

    mul_apple_layer = layer_naive.MulLayer()
    mul_orange_layer = layer_naive.MulLayer()
    add_apple_orange_layer = layer_naive.AddLayer()
    mul_tax_layer = layer_naive.MulLayer()

    apple_price = mul_apple_layer.forward(apple, apple_num)
    orange_price = mul_orange_layer.forward(orange, orange_num)
    all_price = add_apple_orange_layer.forward(apple_price, orange_price)
    price = mul_tax_layer.forward(all_price, tax)

    dprice = 1
    dall_arice, dtax = mul_tax_layer.backword(dprice)
    dapple_price, dorange_price = add_apple_orange_layer.backword(dall_arice)
    dorange, dorange_num = mul_orange_layer.backword(dorange_price)
    dapple, dapple_num = mul_apple_layer.backword(dapple_price)

    print("price:", int(price))
    print("dApple:", dapple)
    print("dApple_num:", int(dapple_num))
    print("dOrange:", dorange)
    print("dOrange_num:", int(dorange_num))
    print("dTax:", dtax)
