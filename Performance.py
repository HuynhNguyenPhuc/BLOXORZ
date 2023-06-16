import time
import psutil
import multiprocessing as mp

def func_running(func, result_queue, before_memory=0, after_memory=0, result=[]):
    process = psutil.Process()
    before_memory = process.memory_info().rss
    result = func()
    after_memory = process.memory_info().rss
    result_queue.put((result, after_memory-before_memory))

def timer(start_time, result_queue, time_limit):
    while result_queue.empty():
        elapsed_time = time.perf_counter() - start_time
        if elapsed_time > time_limit and result_queue.empty():
            result_queue.put("TLE")
            break
    result_queue.put(time.perf_counter() - start_time)

class Performance:
    def __init__(self, func, time_limit=20, except_return=None):
        self.func = func
        self.except_return = except_return
        start_time = time.perf_counter()
        process = psutil.Process()
        process.memory_info().rss
        process.memory_info().rss
        elapsed_time = time.perf_counter() - start_time
        self.time_limit = time_limit + elapsed_time
        self.__init_time_waste = elapsed_time

    def run(self):
        # if __name__ == '__main__':
        result_queue = mp.Queue()
        p1 = mp.Process(target=func_running, args=(self.func, result_queue,))
        p2 = mp.Process(target=timer, args=(time.perf_counter(), result_queue,self.time_limit,))
        p1.start()
        p2.start()
        p2.join()
        p1.terminate()
        result = result_queue.get()
        if result == "TLE":
            with open('Output.txt',mode='a',encoding='utf-8') as f:
                f.writelines("Time Limited Exceeded!\n")
                f.writelines(f"Executing Time: {result_queue.get():.5f} \n")
            print("Time Limited Exceeded!")
            return self.except_return
        else:
            elapsed_time = result_queue.get()
            with open('Output.txt',mode='a',encoding='utf-8') as f:
                f.writelines(f"Executing Time: {elapsed_time:.5f} \n")
                f.writelines(f"Memory Used: {result[1]} bytes \n")
                print(f"Executing Time: {elapsed_time:.5f}")
                print(f"Memory Used: {result[1]} bytes")
            return result[0]