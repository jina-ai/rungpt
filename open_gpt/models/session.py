from collections import OrderedDict


class SessionManager:
    def __init__(self, max_size=100, max_context_length=1024):
        self._session = OrderedDict()
        self.max_size = max_size
        self.max_context_length = max_context_length

    def get(self, session_id):
        return self._session[session_id] if session_id in self._session else None

    def update(self, session_id, past_key_values):
        assert len(past_key_values) > 0, 'past_key_values should not be empty'
        assert len(past_key_values[0]) == 2, (
            'each element in past_key_values ' 'should be a tuple with length of 2'
        )
        assert (
            len(past_key_values[0][0].shape) == 4
        ), 'key / value should be a 4D tensor'

        # past_key_values is a tuple with length of num_layers
        # each layer contains a tuple with length of 2: (key, value)
        # shape of key / value: [batch_size, num_headers, seq_length, dim]
        if past_key_values[0][0].shape[2] > self.max_context_length:
            new_past_key_values = []
            for layers, kv in enumerate(past_key_values):
                new_past_key_values.append(
                    tuple(
                        [
                            kv[0][:, :, -self.max_context_length :, :],
                            kv[1][:, :, -self.max_context_length :, :],
                        ]
                    )
                )
            past_key_values = tuple(new_past_key_values)
        self._session[session_id] = past_key_values
        if self.size > self.max_size:
            self._session.popitem(last=False)

    @property
    def size(self):
        return len(self._session)

    def get_context_length(self, session_id):
        if session_id not in self._session:
            return 0
        return self._session[session_id][0][0].shape[2]

    def clear(self):
        self._session.clear()
