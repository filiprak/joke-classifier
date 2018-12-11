
def is_int(number):
    try:
        int(number)
        return True
    except ValueError:
        return False
