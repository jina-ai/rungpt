from collections import OrderedDict


class SessionManager:
    def __init__(self, max_size=100):
        self._session = OrderedDict()
        self.max_size = max_size

    def get(self, session_id):
        return self._session[session_id] if session_id in self._session else None

    def update(self, session_id, past_key_values):
        self._session[session_id] = past_key_values
        if self.size > self.max_size:
            self._session.popitem(last=False)

    @property
    def size(self):
        return len(self._session)

    def clear(self):
        self._session.clear()
