"""Read current Flickr creator/license declarations as JSON data, never scripts."""
import hashlib
from html.parser import HTMLParser
import json
import re
from urllib.parse import urlsplit


class Metadata(HTMLParser):
    def __init__(self):
        super().__init__(); self.active = False; self.current = []; self.blocks = []; self.title = None

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == 'script' and values.get('type') == 'application/ld+json': self.active = True; self.current = []
        if tag == 'meta' and values.get('property') == 'og:title': self.title = values.get('content')

    def handle_endtag(self, tag):
        if tag == 'script' and self.active:
            self.blocks.append(''.join(self.current)); self.active = False; self.current = []

    def handle_data(self, value):
        if self.active: self.current.append(value)


def image_objects(value):
    if isinstance(value, list):
        for node in value: yield from image_objects(node)
    elif isinstance(value, dict):
        kind = value.get('@type'); kinds = kind if isinstance(kind, list) else [kind]
        if 'ImageObject' in kinds: yield value
        yield from image_objects(value.get('@graph', []))


def photo_page(url):
    parsed = urlsplit(url)
    if parsed.scheme != 'https' or parsed.hostname not in {'flickr.com', 'www.flickr.com'}:
        raise ValueError('Attribution page is not the expected original HTTPS Flickr source')
    match = re.fullmatch(r'/photos/([^/]+)/([0-9]+)/?', parsed.path)
    if not match: raise ValueError('Original page does not identify a single source photo')
    return match.group(1), match.group(2)


def attribution(raw, photo_id, final_url):
    if len(raw) > 2_000_000: raise ValueError('Original attribution page exceeds bounded byte size')
    owner, source_id = photo_page(final_url)
    if not re.fullmatch(r'[0-9]+', str(photo_id)) or source_id != str(photo_id):
        raise ValueError('Original attribution page belongs to another photo')
    parser = Metadata(); parser.feed(raw.decode('utf-8')); parser.close(); declarations = []
    for block in parser.blocks:
        for node in image_objects(json.loads(block)):
            page = node.get('acquireLicensePage')
            if not isinstance(page, str): continue
            node_owner, node_id = photo_page(page)
            if node_id != str(photo_id): continue
            if node_owner != owner: raise ValueError('Current source owner differs between original page and image declaration')
            license_url = urlsplit(node.get('license', ''))
            if (license_url.scheme not in {'http', 'https'} or license_url.hostname != 'creativecommons.org'
                    or license_url.path.rstrip('/') != '/licenses/by/2.0'):
                raise ValueError('Current original photo is not declared CC-BY2.0')
            author = node.get('author')
            if not isinstance(author, dict) or not isinstance(author.get('name'), str) or not author['name'].strip():
                raise ValueError('Original photo creator attribution absent')
            author_url = urlsplit(author.get('url', ''))
            if (author_url.scheme != 'https' or author_url.hostname not in {'flickr.com', 'www.flickr.com'}
                    or author_url.path.rstrip('/') != '/photos/' + owner):
                raise ValueError('Original creator profile does not match photo owner')
            content = urlsplit(node.get('contentUrl', ''))
            if (not content.hostname or not content.hostname.endswith('.staticflickr.com')
                    or not re.match(str(photo_id) + r'_', content.path.rsplit('/', 1)[-1])):
                raise ValueError('Original declared image belongs to another photo')
            creator = node.get('creator')
            if creator is not None and (not isinstance(creator, dict) or creator.get('name') != author['name']
                                        or creator.get('url') != author['url']):
                raise ValueError('Conflicting original creator declarations')
            declarations.append({'source_flickr_photo_id': str(photo_id), 'author': author['name'],
                                 'author_profile_url': author['url'], 'source_url': page,
                                 'license': 'CC-BY-2.0', 'license_url': 'https://creativecommons.org/licenses/by/2.0/',
                                 'current_source_image_url': node['contentUrl']})
    if not declarations or any(value != declarations[0] for value in declarations):
        raise ValueError('Missing or conflicting original licensed-image declarations')
    return {**declarations[0], 'source_title': parser.title, 'attribution_html_sha256': hashlib.sha256(raw).hexdigest(),
            'attribution_origin': 'Original current Flickr ImageObject/author JSON-LD declaration; no external model.'}
