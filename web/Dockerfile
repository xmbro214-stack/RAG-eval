FROM node:20-bookworm-slim AS build

ARG NPM_REGISTRY=https://registry.npmjs.org/

WORKDIR /app/web

COPY web/package.json web/package-lock.json ./
RUN npm config set registry "$NPM_REGISTRY" \
    && npm ci --no-audit --fund=false

COPY web/ ./
RUN npm run build

FROM nginx@sha256:6e23479198b998e5e25921dff8455837c7636a67111a04a635cf1bb363d199dc

COPY web/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/web/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://127.0.0.1/ >/dev/null || exit 1
