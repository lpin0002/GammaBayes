import time, multiprocessing
from multiprocessing import Pool




def do_something(times):
    print(f'Sleeping {times} second(s)...')
    time.sleep(times)
    return 'Done Sleeping for {times} second(s).'


if __name__ == '__main__':
    amountoftimeyouarewillingtowait = 10

    times = list(range(1, amountoftimeyouarewillingtowait+1))
    times.reverse()
    print(times)
    start = time.perf_counter()
    num_cores = multiprocessing.cpu_count()
    with Pool(num_cores) as pool:
        for result in pool.imap(do_something, times):
            print(result)

        pool.close()    

    finish = time.perf_counter()

    print(f"Finished in {finish-start:.2f} second(s)")