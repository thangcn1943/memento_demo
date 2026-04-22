def stock_analysis(stock_name):
    if stock_name in ["VCB", "VPB", "TCB"]:
        return f"{stock_name} is a good stock to invest in."
    elif stock_name in ["FLC", "ROS"]:
        return f"{stock_name} is a risky stock to invest in."
    elif stock_name in ["VIC", "VHM"]:
        return f"{stock_name} is a stable stock to invest in."
    else:
        return f"Sorry, I don't have information about {stock_name}."
    
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: analyze.py <stock_name>")
        sys.exit(1)

    stock_name = sys.argv[1]
    result = stock_analysis(stock_name)
    print(result)