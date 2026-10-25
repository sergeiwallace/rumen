"""Offline SSRF regressions through the RSS and podcast entry points."""

from __future__ import annotations

import io
import socket
import ssl
from collections import Counter
from email.parser import Parser
from pathlib import Path

import pytest

from rumen.adapters.base import StatefulFetchError
from rumen.adapters.rss_atom import RssAtomAdapter
from rumen.podcasts.cache import initialize
from rumen.podcasts.http import PublisherHttpClient
from rumen.podcasts.sync import synchronize

_FIXTURES = Path(__file__).parents[1] / "fixtures"
_FEED_URL = "https://publisher.example/feed"
_TRANSCRIPT_URL = "https://publisher.example/one.vtt"
_AUDIO_URL = "https://publisher.example/audio-one.mp3"
_RSS = (_FIXTURES / "rss_atom" / "representative.rss").read_bytes()
_PODCAST = (_FIXTURES / "spotify_podcasts" / "feed.xml").read_bytes()
_VTT = (_FIXTURES / "spotify_podcasts" / "transcript.vtt").read_bytes()
_UNSAFE_URLS = [
    "http://publisher.example/text",
    "https://user:password@publisher.example/text",
    "https://@publisher.example/text",
    "https://127.0.0.1:8080/admin",
    "https://10.0.0.1/text",
    "https://172.16.0.1/text",
    "https://192.168.1.1/text",
    "https://169.254.169.254/text",
    "https://[::1]/text",
    "https://[fe80::1]/text",
    "https://[fd00::1]/text",
    "https://private.example/text",
    "https://mixed.example/text",
]


@pytest.fixture
def wire_requests():
    return []


@pytest.fixture
def network(monkeypatch, wire_requests):
    """Fake only DNS, TCP sockets and TLS wrapping; run real HTTP framing."""
    routes = {}
    connections = []
    requests = []
    tls = []
    lookups = Counter()
    answers = {
        "private.example": ["127.0.0.1"],
        "mixed.example": ["8.8.8.8", "10.0.0.1"],
    }
    rebinding = set()

    def resolve(host, port, *args, **kwargs):
        lookups[host] += 1
        addresses = answers.get(host, ["8.8.8.8"])
        if host in rebinding and lookups[host] > 1:
            addresses = ["127.0.0.1"]
        return [
            (
                socket.AF_INET6 if ":" in address else socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                (address, port or 443),
            )
            for address in addresses
        ]

    class FakeSocket:
        def __init__(self, *args, **kwargs):
            self.sent = b""
            self.timeout = None

        def settimeout(self, timeout):
            self.timeout = timeout

        def gettimeout(self):
            return self.timeout

        def setsockopt(self, *args):
            pass

        def connect(self, address):
            connections.append(address)

        def sendall(self, data):
            self.sent += data

        def makefile(self, *args, **kwargs):
            request = self.sent.decode("ascii")
            wire_requests.append(request)
            path = request.split(" ")[1]
            host = next(
                line.split(": ", 1)[1] for line in request.split("\r\n") if line.startswith("Host:")
            )
            requests.append((host, path))
            status, headers, body = routes.get((host, path), (200, {}, _VTT))
            head = f"HTTP/1.1 {status} Response\r\nContent-Length: {len(body)}\r\n"
            head += "".join(f"{key}: {value}\r\n" for key, value in headers.items())
            return io.BytesIO(head.encode("ascii") + b"\r\n" + body)

        def close(self):
            pass

    def wrap(context, sock, *, server_hostname=None, **kwargs):
        tls.append((server_hostname, context.check_hostname, context.verify_mode))
        return sock

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(socket, "socket", FakeSocket)
    monkeypatch.setattr(ssl.SSLContext, "wrap_socket", wrap)
    # The production transport must bypass proxy routing. Keep the fixture offline
    # even if the test runner has proxy environment settings.
    monkeypatch.setattr("urllib.request.getproxies", lambda: {})
    return routes, connections, requests, tls, lookups, answers, rebinding


def _sync(tmp_path, *, client=None):
    return synchronize(
        {
            "cache_path": str(tmp_path / "cache.db"),
            "exports": [str(_FIXTURES / "spotify_podcasts" / "history.json")],
            "feeds": [
                {
                    "show": "Test Show",
                    "url": _FEED_URL,
                    "allow_asr": True,
                    "asr": {
                        "argv": ["unused", "{input}", "{output}"],
                        "tool_model_identifier": "test",
                    },
                }
            ],
        },
        client=client or PublisherHttpClient(max_attempts=1, request_interval_seconds=0),
    )


@pytest.mark.parametrize("url", _UNSAFE_URLS)
@pytest.mark.parametrize("resource", ["transcript", "enclosure"])
def test_given_unsafe_publisher_url_when_syncing_then_never_connects_or_stores_text(
    tmp_path, monkeypatch, network, url, resource
):
    routes, connections, requests, *_ = network
    feed = _PODCAST.replace(_TRANSCRIPT_URL.encode(), url.encode())
    if resource == "enclosure":
        feed = _PODCAST.replace(_AUDIO_URL.encode(), url.encode()).replace(
            b'<podcast:transcript url="https://publisher.example/one.vtt" type="text/vtt" language="en"/>',
            b"",
        )
    routes[("publisher.example", "/feed")] = (200, {}, feed)
    # If an enclosure is fetched, a downstream ASR success must not hide it.
    monkeypatch.setattr("rumen.podcasts.sync.transcribe", lambda *_: pytest.fail("ASR ran"))
    # Transcript rejection should not fall back to the valid enclosure in this case.
    if resource == "transcript":
        routes[("publisher.example", "/audio-one.mp3")] = (403, {}, b"")
    result = _sync(tmp_path)
    assert result["failure"] >= 1
    assert result["publisher_transcript"] == result["asr"] == 0
    assert all(address[0] == "8.8.8.8" for address in connections)
    allowed = {("publisher.example", "/feed"), ("publisher.example", "/audio-one.mp3")}
    assert set(requests) <= allowed
    conn = initialize(tmp_path / "cache.db")
    assert conn.execute("SELECT COUNT(*) FROM podcast_transcripts").fetchone()[0] == 0
    conn.close()


@pytest.mark.parametrize("url", _UNSAFE_URLS)
def test_given_unsafe_feed_url_when_fetching_then_rejects_before_connection(network, url):
    with pytest.raises(StatefulFetchError) as error:
        RssAtomAdapter("test", url).fetch_new(None)
    assert error.value.diagnostics["reason"] == "unsafe_target"
    assert network[1] == []


@pytest.mark.parametrize("connector", ["rss", "podcast"])
@pytest.mark.parametrize("url", _UNSAFE_URLS)
def test_given_unsafe_redirect_when_fetching_then_rejects_before_following(
    tmp_path, network, connector, url
):
    routes, connections, requests, *_ = network
    if connector == "rss":
        routes[("publisher.example", "/feed")] = (302, {"Location": url}, b"")
        with pytest.raises(StatefulFetchError):
            RssAtomAdapter("test", _FEED_URL).fetch_new(None)
        assert requests == [("publisher.example", "/feed")]
    else:
        routes[("publisher.example", "/feed")] = (200, {}, _PODCAST)
        routes[("publisher.example", "/one.vtt")] = (302, {"Location": url}, b"")
        routes[("publisher.example", "/audio-one.mp3")] = (403, {}, b"")
        result = _sync(tmp_path)
        assert result["publisher_transcript"] == 0
        assert result["failure"] >= 1
        assert all(host == "publisher.example" for host, _ in requests)
    assert all(address[0] == "8.8.8.8" for address in connections)


@pytest.mark.parametrize("connector", ["rss", "podcast"])
@pytest.mark.parametrize("redirect", [False, True])
def test_given_rebinding_dns_when_fetching_then_pins_public_ip_and_verifies_hostname(
    tmp_path, network, connector, redirect
):
    routes, connections, requests, tls, lookups, _, rebinding = network
    rebinding.update({"publisher.example", "cdn.example"})
    routes[("publisher.example", "/feed")] = (200, {}, _RSS if connector == "rss" else _PODCAST)
    if redirect:
        if connector == "rss":
            routes[("publisher.example", "/feed")] = (
                302,
                {"Location": "https://cdn.example:8443/feed?version=1"},
                b"",
            )
            routes[("cdn.example:8443", "/feed?version=1")] = (200, {}, _RSS)
        else:
            routes[("publisher.example", "/one.vtt")] = (
                302,
                {"Location": "https://cdn.example:8443/text?version=1"},
                b"",
            )
            routes[("cdn.example:8443", "/text?version=1")] = (200, {}, _VTT)
    if connector == "rss":
        result = RssAtomAdapter("test", _FEED_URL).fetch_new(None)
        assert len(result) == 1
        assert lookups["publisher.example"] == 1
    else:
        # A new request may re-resolve the feed hostname. Use an independent
        # transcript hostname so the probe changes DNS within a single request.
        routes[("publisher.example", "/feed")] = (
            200,
            {},
            _PODCAST.replace(_TRANSCRIPT_URL.encode(), b"https://text.example/one.vtt"),
        )
        rebinding.add("text.example")
        if redirect:
            routes[("text.example", "/one.vtt")] = routes[("publisher.example", "/one.vtt")]
        result = _sync(tmp_path)
        assert result["publisher_transcript"] == 1
        assert lookups["text.example"] == 1
    assert connections
    assert all(address[0] == "8.8.8.8" for address in connections)
    assert all(check and mode == ssl.CERT_REQUIRED for _, check, mode in tls)
    assert len(tls) == len(connections)
    assert [host for host, *_ in tls] == [host.split(":")[0] for host, _ in requests]
    if redirect:
        assert ("8.8.8.8", 8443) in connections
        assert lookups["cdn.example"] == 1


def test_given_resolver_seam_when_rss_fetches_then_never_uses_transport_dns(network):
    routes, connections, _, tls, lookups, answers, _ = network
    answers["publisher.example"] = ["127.0.0.1"]
    routes[("publisher.example", "/feed")] = (200, {}, _RSS)
    adapter = RssAtomAdapter("test", _FEED_URL, resolver=lambda _host: ["8.8.4.4"])

    assert len(adapter.fetch_new(None)) == 1
    assert connections == [("8.8.4.4", 443)]
    assert lookups == {}
    assert tls == [("publisher.example", True, ssl.CERT_REQUIRED)]


@pytest.mark.parametrize("connector", ["rss", "podcast"])
def test_given_certificate_error_when_fetching_then_rejects_response(
    tmp_path, monkeypatch, network, connector
):
    def reject_certificate(*args, **kwargs):
        raise ssl.SSLCertVerificationError("certificate does not match hostname")

    monkeypatch.setattr(ssl.SSLContext, "wrap_socket", reject_certificate)
    if connector == "rss":
        with pytest.raises(StatefulFetchError) as error:
            RssAtomAdapter("test", _FEED_URL).fetch_new(None)
        assert error.value.diagnostics == {"reason": "transport_failure"}
    else:
        result = _sync(tmp_path)
        assert result["failure"] == 1
        assert result["publisher_transcript"] == 0
    assert network[2] == []


@pytest.mark.parametrize("connector", ["rss", "podcast"])
@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_given_public_relative_redirect_when_fetching_then_follows_with_validated_connection(
    tmp_path, network, connector, status
):
    routes, connections, requests, tls, *_ = network
    routes[("publisher.example", "/feed")] = (status, {"Location": "/final"}, b"")
    routes[("publisher.example", "/final")] = (200, {}, _RSS if connector == "rss" else _PODCAST)
    if connector == "rss":
        assert len(RssAtomAdapter("test", _FEED_URL).fetch_new(None)) == 1
    else:
        assert _sync(tmp_path)["publisher_transcript"] == 1
    assert requests[:2] == [("publisher.example", "/feed"), ("publisher.example", "/final")]
    assert all(address == ("8.8.8.8", 443) for address in connections)
    assert all(row == ("publisher.example", True, ssl.CERT_REQUIRED) for row in tls)


@pytest.mark.parametrize("connector", ["rss", "podcast"])
@pytest.mark.parametrize("headers", [{}, {"Location": "/feed"}])
def test_given_broken_or_looping_redirect_when_fetching_then_fails_with_bounded_requests(
    tmp_path, network, connector, headers
):
    routes, connections, *_ = network
    routes[("publisher.example", "/feed")] = (302, headers, b"")
    if connector == "rss":
        with pytest.raises(StatefulFetchError) as error:
            RssAtomAdapter("test", _FEED_URL).fetch_new(None)
        assert error.value.diagnostics["reason"] == (
            "redirect_limit" if headers else "redirect_failure"
        )
    else:
        assert _sync(tmp_path)["failure"] == 1
    assert len(connections) == (6 if headers else 1)


@pytest.mark.parametrize(
    "module",
    [
        "github_releases",
        "github_repository",
        "github_stars",
        "hacker_news",
        "hf_daily_papers",
        "npm_registry",
    ],
)
def test_given_service_redirect_to_private_host_when_fetching_then_no_private_connection(
    network, module
):
    import importlib
    from urllib.request import Request

    from rumen.net_safety import UnsafeTarget

    routes, connections, requests, *_ = network
    routes[("provider.example", "/resource")] = (302, {"Location": "https://127.0.0.1/admin"}, b"")
    source = "https://provider.example/resource"
    if module in {"github_repository", "github_stars"}:
        source = Request(source)
    with pytest.raises(UnsafeTarget):
        importlib.import_module(f"rumen.adapters.{module}")._http_get(source)
    assert connections == [("8.8.8.8", 443)]
    assert requests == [("provider.example", "/resource")]


def test_given_feed_body_that_names_local_file_when_parsed_then_file_is_not_read(tmp_path):
    path = tmp_path / "private.rss"
    path.write_bytes(_RSS)
    with pytest.raises(StatefulFetchError, match="unrecognized"):
        RssAtomAdapter("example", _FEED_URL)._parse(str(path).encode(), _FEED_URL)


def test_given_service_post_when_sent_then_body_method_and_headers_are_preserved(
    network, monkeypatch
):
    from urllib.request import Request

    from rumen.adapters.github_stars import _http_get
    from rumen.net_safety import _PinnedHTTPSConnection

    sent = []
    original = _PinnedHTTPSConnection.request

    def record(connection, method, path, body=None, headers=None):
        sent.append((method, body, headers))
        return original(connection, method, path, body=body, headers=headers)

    monkeypatch.setattr(_PinnedHTTPSConnection, "request", record)
    routes, connections, requests, *_ = network
    routes[("provider.example", "/graphql")] = (200, {}, b'{"data": {}}')
    assert (
        _http_get(
            Request(
                "https://provider.example/graphql",
                data=b'{"query": "example"}',
                headers={"Content-Type": "application/json"},
                method="POST",
            )
        )
        == '{"data": {}}'
    )
    assert connections == [("8.8.8.8", 443)]
    assert requests == [("provider.example", "/graphql")]
    assert len(sent) == 1
    method, body, headers = sent[0]
    assert (method, body) == ("POST", b'{"query": "example"}')
    assert headers["Content-type"] == "application/json"


@pytest.mark.parametrize("connector", ["service", "publisher", "rss"])
@pytest.mark.parametrize("header_name", [None, "User-Agent", "user-agent", "uSeR-aGeNt"])
def test_given_user_agent_headers_when_fetching_then_wire_has_one_default_or_caller_value(
    network, wire_requests, connector, header_name
):
    from urllib.request import Request

    from rumen import __version__
    from rumen.adapters.rss_atom import _default_transport
    from rumen.net_safety import open_public_request

    routes, *_ = network
    routes[("provider.example", "/resource")] = (302, {"Location": "/final"}, b"")
    routes[("provider.example", "/final")] = (200, {}, b"response")
    headers = {header_name: "example-client/2.0"} if header_name else {}
    original_headers = headers.copy()
    url = "https://provider.example/resource"
    if connector == "service":
        with open_public_request(Request(url, headers=headers)) as response:
            assert response.read() == b"response"
    elif connector == "publisher":
        response = PublisherHttpClient(max_attempts=1, request_interval_seconds=0).get(
            url, headers=headers, byte_ceiling=100
        )
        assert response.body == b"response"
    else:
        assert _default_transport(url, headers, 30, lambda _: ["8.8.8.8"]).body == b"response"

    expected = (
        "example-client/2.0"
        if header_name
        else ("rumen/rss-atom" if connector == "rss" else f"rumen/{__version__}")
    )
    assert len(wire_requests) == 2
    for request in wire_requests:
        framed_headers = Parser().parsestr(request.split("\r\n", 1)[1])
        assert framed_headers.get_all("User-Agent") == [expected]
    assert headers == original_headers


def test_given_service_error_when_returned_then_status_and_body_remain_readable(network):
    from rumen.adapters.youtube import _default_transport

    routes, *_ = network
    routes[("www.googleapis.com", "/youtube/v3/videos?")] = (429, {"Retry-After": "10"}, b"{}")
    # An empty query is normalized to the path without the question mark.
    routes[("www.googleapis.com", "/youtube/v3/videos")] = (429, {"Retry-After": "10"}, b"{}")
    status, headers, body = _default_transport("videos", {})
    assert status == 429
    assert headers["Retry-After"] == "10"
    assert body == b"{}"


def test_given_wikidata_redirect_to_private_host_when_resolved_then_returns_none(network):
    from rumen.taxonomy.donors.wikidata_link import fetch_wikidata_entity

    routes, connections, requests, *_ = network
    routes[("www.wikidata.org", "/wiki/Special:EntityData/Q1.json")] = (
        302,
        {"Location": "https://127.0.0.1/admin"},
        b"",
    )
    assert fetch_wikidata_entity("Q1") is None
    assert connections == [("8.8.8.8", 443)]
    assert requests == [("www.wikidata.org", "/wiki/Special:EntityData/Q1.json")]
