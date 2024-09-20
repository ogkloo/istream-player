import zmq
import argparse

from time import sleep

def check_messages(socket, ack_sender):
    while True:
        message = None
        try:
            message = socket.recv_string(flags=zmq.NOBLOCK)
        except zmq.Again:
            # No message, continue with other tasks
            pass

        if message is not None:
            try:
                sleep(0.2)
                print(f"acked {message}")
                ack_sender.send_string("ack")
            except:
                print("Ack send failed")
        sleep(0.1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sends an event message over ZMQ.')

    port: int = 9876
    ack_port: int = 44444

    # Setup ZMQ context
    context = zmq.Context()

    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://localhost:{port}")
    socket.setsockopt_string(zmq.SUBSCRIBE, "evs")

    ack_sender = context.socket(zmq.PUB)
    ack_sender.bind(f"tcp://localhost:{ack_port}")

    check_messages(socket, ack_sender)



