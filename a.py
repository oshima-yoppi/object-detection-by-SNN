import time

for i in range(10):
    try:
        print(i)
        time.sleep(1)
    except KeyboardInterrupt:
        print('Ctrl+C')
        break