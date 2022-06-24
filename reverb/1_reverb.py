import reverb

test_server = reverb.Server(tables=[
    reverb.Table.queue(name='my_queue', max_size=1000)]
)


client = reverb.Client(f'localhost:{test_server.port}')
# print(client.server_info())

# necessary to provide specs parameter for server to verify the specs of the given data in table
