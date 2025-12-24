FROM node:18

WORKDIR /app

COPY package.json pnpm-lock.json /app/

COPY . /app

RUN npm install

CMD ["npm", "start"]
