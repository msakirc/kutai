// k6 smoke script — replaced at run-time by the synthetic_check executor.
// Default: hit TARGET_URL with VUS virtual users for DURATION, record latency.
import http from 'k6/http';
import { sleep } from 'k6';

export const options = {
  vus: __ENV.VUS ? parseInt(__ENV.VUS, 10) : 5,
  duration: __ENV.DURATION || '30s',
};

export default function () {
  http.get(__ENV.TARGET_URL || 'http://localhost');
  sleep(1);
}
