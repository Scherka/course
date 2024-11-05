symbols = list(' abcdefghijklmnopqrstuvwxyz.,()-')
symbol_dict = {symbol: i for i, symbol in enumerate(symbols)}
inverted_dict = {v: k for k, v in symbol_dict.items()}
def code(mes):
    bin_str =''
    for c in mes:
        bin_str += f'{symbol_dict[c.lower()]:05b}'
    return bin_str

def decode(bin_str):
    mes = ''
    i = 0
    while i < len(bin_str):
        mes += inverted_dict[int(bin_str[i:i+5],2)]
        i+=5
    return mes